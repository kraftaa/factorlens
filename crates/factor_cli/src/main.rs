use anyhow::{anyhow, Result};
use chrono::{Datelike, Weekday};
use clap::{Parser, Subcommand, ValueEnum};
use csv::StringRecord;
use factor_core::{
    compute_outliers, fit_pca, portfolio_factor_contributions, portfolio_returns_from_weights,
    PortfolioWeight,
};
use factor_io::{
    list_artifact_paths, read_artifact_summary, read_portfolio_csv, read_prices_csv,
    write_factor_artifacts,
};
use llm_local::{build_client, Backend};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "factorlens")]
#[command(about = "Local factor attribution + explainability CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Factors {
        #[command(subcommand)]
        command: FactorsCommand,
    },
    Explain {
        #[arg(long, value_enum, default_value = "local")]
        backend: BackendArg,
        #[arg(long)]
        model: String,
        #[arg(long)]
        artifacts: PathBuf,
        #[arg(long)]
        question: String,
        #[arg(long, value_delimiter = ',')]
        focus_factors: Vec<String>,
        #[arg(long)]
        factor_labels: Option<PathBuf>,
    },
    Report {
        #[arg(long)]
        artifacts: PathBuf,
        #[arg(long, default_value = "markdown")]
        format: String,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum FactorsCommand {
    Fit {
        #[arg(long)]
        prices: PathBuf,
        #[arg(long, default_value_t = 5)]
        k: usize,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        portfolio: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        include_weekends: bool,
    },
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum BackendArg {
    Local,
    Bedrock,
}

#[derive(Debug, Clone)]
struct AttributionInsight {
    date: String,
    total: f64,
    factors: Vec<(String, f64)>,
}

type FactorLabels = HashMap<String, String>;

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Factors {
            command:
                FactorsCommand::Fit {
                    prices,
                    k,
                    out,
                    portfolio,
                    include_weekends,
                },
        } => {
            let mut prices = read_prices_csv(&prices)?;
            if !include_weekends {
                prices.retain(|p| !matches!(p.date.weekday(), Weekday::Sat | Weekday::Sun));
            }

            let model = fit_pca(&prices, k)?;

            let weights = if let Some(path) = portfolio {
                read_portfolio_csv(&path)?
            } else {
                equal_weights(&model.tickers)
            };

            let portfolio_returns = portfolio_returns_from_weights(&model, &weights)?;
            let contributions = portfolio_factor_contributions(&model, &weights)?;
            let outliers = compute_outliers(&model.dates, &portfolio_returns, &contributions, 15);
            write_factor_artifacts(&out, &model, &contributions, &outliers)?;

            println!("Wrote artifacts to {}", out.display());
            println!(
                "Weekend rows: {}",
                if include_weekends {
                    "included"
                } else {
                    "excluded (default)"
                }
            );
            for p in list_artifact_paths(&out) {
                println!("- {}", p.display());
            }
        }
        Commands::Explain {
            backend,
            model,
            artifacts,
            question,
            focus_factors,
            factor_labels,
        } => {
            let summary = read_artifact_summary(&artifacts)?;
            let labels = load_factor_labels(factor_labels.as_ref())?;
            let insight = attribution_worst_day(&artifacts, &focus_factors)?;

            if let Some(answer) = deterministic_answer(&question, insight.as_ref(), &labels) {
                println!("{}", answer);
                return Ok(());
            }

            let backend = match backend {
                BackendArg::Local => Backend::Local,
                BackendArg::Bedrock => Backend::Bedrock,
            };
            let client = build_client(backend, model);

            let context = build_prompt_context(
                &summary,
                &artifacts,
                &focus_factors,
                insight.as_ref(),
                &labels,
            )?;
            let system = "You are a risk analysis assistant. Use only the provided artifact context. If data is missing, say unknown. Respond in plain text only. Do not output code, role tags, or tool instructions. Be specific with dates and factor contributions when available.";
            let user = format!("Question: {}\n\nArtifact context:\n{}", question, context);
            let answer = client.answer(system, &user)?;

            println!("{}", answer);
        }
        Commands::Report {
            artifacts,
            format,
            out,
        } => {
            if format.to_lowercase() != "markdown" {
                return Err(anyhow!("only markdown format is supported in MVP"));
            }
            let summary = read_artifact_summary(&artifacts)?;
            let md = report::markdown_report(&summary)?;
            fs::write(&out, md)?;
            println!("Report written to {}", out.display());
        }
    }

    Ok(())
}

fn equal_weights(tickers: &[String]) -> Vec<PortfolioWeight> {
    let w = 1.0 / tickers.len() as f64;
    tickers
        .iter()
        .map(|t| PortfolioWeight {
            ticker: t.clone(),
            weight: w,
        })
        .collect()
}

fn deterministic_answer(
    question: &str,
    insight: Option<&AttributionInsight>,
    labels: &FactorLabels,
) -> Option<String> {
    let q = question.to_lowercase();
    if !(q.contains("drawdown") || q.contains("dropped") || q.contains("drop")) {
        return None;
    }

    let insight = insight?;
    if insight.factors.is_empty() {
        return Some("Unable to isolate factor drivers from attribution.csv.".to_string());
    }

    let top = &insight.factors[0];
    let pct = if insight.total.abs() > 1e-12 {
        (top.1 / insight.total) * 100.0
    } else {
        0.0
    };

    let others = insight
        .factors
        .iter()
        .skip(1)
        .take(3)
        .map(|(f, v)| format!("{}={:.6}", factor_with_label(f, labels), v))
        .collect::<Vec<_>>()
        .join(", ");

    Some(format!(
        "Largest modeled drawdown day is {} with total factor contribution {:.6}. Primary driver is {} at {:.6} ({:.1}% of total). Other selected factor contributions: {}.",
        insight.date,
        insight.total,
        factor_with_label(&top.0, labels),
        top.1,
        pct,
        if others.is_empty() {
            "none".to_string()
        } else {
            others
        }
    ))
}

fn build_prompt_context(
    summary: &factor_io::ArtifactSummary,
    artifacts: &PathBuf,
    focus_factors: &[String],
    insight: Option<&AttributionInsight>,
    labels: &FactorLabels,
) -> Result<String> {
    let top_var = summary
        .model
        .explained_variance_ratio
        .iter()
        .take(5)
        .enumerate()
        .map(|(i, v)| format!("factor_{}: {:.4}", i + 1, v))
        .collect::<Vec<_>>()
        .join(", ");

    let outliers = summary
        .top_outliers
        .iter()
        .take(10)
        .map(|o| format!("{} residual {:.5}", o.date, o.residual))
        .collect::<Vec<_>>()
        .join("; ");

    let focus_line = if focus_factors.is_empty() {
        "focus_factors=all".to_string()
    } else {
        format!("focus_factors={}", focus_factors.join(","))
    };

    let drawdown_line = if let Some(i) = insight {
        let factor_line = i
            .factors
            .iter()
            .map(|(name, v)| format!("{}={:.6}", factor_with_label(name, labels), v))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "worst_day_total_factor_contrib: date={} value={:.6} breakdown: {}",
            i.date, i.total, factor_line
        )
    } else {
        "worst_day_total_factor_contrib=unknown".to_string()
    };

    let snippet = format!(
        "k={} | assets={} | observations={}\nexplained_variance={}\noutliers={}\n{}\nartifacts_dir={}",
        summary.model.k,
        summary.model.tickers.len(),
        summary.model.dates.len(),
        top_var,
        outliers,
        focus_line,
        artifacts.display()
    );
    Ok(format!("{}\n{}", snippet, drawdown_line))
}

fn attribution_worst_day(
    artifacts: &PathBuf,
    focus_factors: &[String],
) -> Result<Option<AttributionInsight>> {
    let path = artifacts.join("attribution.csv");
    if !path.exists() {
        return Ok(None);
    }

    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let total_idx = headers
        .iter()
        .position(|h| h == "total_factor_contrib")
        .ok_or_else(|| anyhow!("attribution.csv missing total_factor_contrib column"))?;
    let date_idx = headers
        .iter()
        .position(|h| h == "date")
        .ok_or_else(|| anyhow!("attribution.csv missing date column"))?;

    let wanted = focus_factors
        .iter()
        .map(|f| normalize_factor_name(f))
        .collect::<Vec<_>>();

    let factor_cols = headers
        .iter()
        .enumerate()
        .filter_map(|(i, h)| {
            if h.starts_with("factor_") && h.ends_with("_contrib") {
                if wanted.is_empty() || wanted.iter().any(|w| h == w) {
                    Some((i, h.to_string()))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let mut worst: Option<(String, f64, Vec<(String, f64)>)> = None;
    for row in rdr.records() {
        let row = row?;
        let date = row_get(&row, date_idx)?.to_string();
        let total = row_get(&row, total_idx)?.parse::<f64>().unwrap_or(0.0);

        let mut factors = factor_cols
            .iter()
            .map(|(idx, name)| {
                let v = row_get(&row, *idx)
                    .ok()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                (name.clone(), v)
            })
            .collect::<Vec<_>>();

        factors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        match &worst {
            None => worst = Some((date, total, factors)),
            Some((_, current, _)) if total < *current => worst = Some((date, total, factors)),
            _ => {}
        }
    }

    Ok(worst.map(|(date, total, factors)| AttributionInsight {
        date,
        total,
        factors,
    }))
}

fn normalize_factor_name(name: &str) -> String {
    let n = name.trim();
    if n.starts_with("factor_") && n.ends_with("_contrib") {
        n.to_string()
    } else if n.starts_with("factor_") {
        format!("{}_contrib", n)
    } else {
        format!("factor_{}_contrib", n)
    }
}

fn factor_with_label(raw: &str, labels: &FactorLabels) -> String {
    let normalized = normalize_factor_name(raw);
    if let Some(custom) = labels.get(&normalized) {
        format!("{} ({})", normalized, custom)
    } else {
        format!("{} ({})", normalized, factor_normal_name(raw))
    }
}

fn factor_normal_name(raw: &str) -> &'static str {
    match factor_index(raw) {
        Some(1) => "Broad Market Move",
        Some(2) => "Sector/Style Rotation",
        Some(3) => "Cross-Sectional Spread",
        Some(4) => "Secondary Regime Shift",
        Some(5) => "Residual Risk Cluster",
        _ => "Latent Statistical Factor",
    }
}

fn factor_index(raw: &str) -> Option<usize> {
    let s = raw.trim();
    let s = s.strip_prefix("factor_")?;
    let s = s.strip_suffix("_contrib").unwrap_or(s);
    s.parse::<usize>().ok()
}

fn row_get<'a>(row: &'a StringRecord, idx: usize) -> Result<&'a str> {
    row.get(idx)
        .ok_or_else(|| anyhow!("invalid csv index {}", idx))
}

fn load_factor_labels(path: Option<&PathBuf>) -> Result<FactorLabels> {
    let mut out = HashMap::new();
    let Some(path) = path else {
        return Ok(out);
    };

    let content = fs::read_to_string(path)?;
    for (i, line) in content.lines().enumerate() {
        let raw = line.trim();
        if raw.is_empty() || raw.starts_with('#') {
            continue;
        }
        let lower = raw.to_lowercase();
        if i == 0 && (lower == "factor,label" || lower == "factor_name,label") {
            continue;
        }

        let (factor_raw, label_raw) = if let Some((a, b)) = raw.split_once(',') {
            (a.trim(), b.trim())
        } else if let Some((a, b)) = raw.split_once('\t') {
            (a.trim(), b.trim())
        } else {
            return Err(anyhow!(
                "invalid factor label line '{}'; expected factor,label",
                raw
            ));
        };

        if factor_raw.is_empty() || label_raw.is_empty() {
            return Err(anyhow!(
                "invalid factor label line '{}'; factor and label are required",
                raw
            ));
        }
        out.insert(normalize_factor_name(factor_raw), label_raw.to_string());
    }

    Ok(out)
}
