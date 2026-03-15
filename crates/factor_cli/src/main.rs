use anyhow::{anyhow, Result};
use chrono::{Datelike, Weekday};
use clap::{Args, Parser, Subcommand, ValueEnum};
use csv::StringRecord;
use factor_core::{
    compute_outliers, fit_pca, portfolio_factor_contributions, portfolio_returns_from_weights,
    PortfolioWeight,
};
use factor_io::{
    list_artifact_paths, read_artifact_summary, read_holdings_as_weights_csv, read_portfolio_csv,
    read_prices_csv, write_factor_artifacts,
};
use llm_local::{build_client, Backend};
use nalgebra::{DMatrix, DVector};
use postgres::{Client, NoTls};
use postgres_rustls::MakeTlsConnector;
use pulldown_cmark::{html, Options, Parser as MdParser};
use rustls::ClientConfig;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::BufReader;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Parser)]
#[command(name = "factorlens")]
#[command(about = "Local factor attribution + explainability CLI")]
#[command(version)]
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
    ExplainAnalyze {
        #[arg(long, value_enum, default_value = "local")]
        backend: BackendArg,
        #[arg(long)]
        model: String,
        #[arg(long)]
        analysis_json: PathBuf,
        #[arg(long)]
        question: String,
    },
    Report {
        #[arg(long)]
        artifacts: PathBuf,
        #[arg(long, default_value = "markdown")]
        format: String,
        #[arg(long)]
        out: PathBuf,
    },
    Analyze(AnalyzeArgs),
    AnalyzePeriod(AnalyzeArgs),
    AnalyzeValidate(AnalyzeArgs),
    AnalyzeInvestigate(InvestigateArgs),
    AnalyzeDrivers(AnalyzeDriversArgs),
    AnalyzeSuggest(AnalyzeSuggestArgs),
    AnalyzeCompare(AnalyzeCompareArgs),
}

#[derive(Subcommand)]
enum FactorsCommand {
    Fit {
        #[arg(long)]
        prices: PathBuf,
        #[arg(long, default_value_t = 5)]
        k: usize,
        #[arg(long, default_value_t = false)]
        k_auto: bool,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        portfolio: Option<PathBuf>,
        #[arg(long)]
        holdings: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        include_weekends: bool,
    },
    Regress {
        #[arg(long)]
        prices: PathBuf,
        #[arg(long)]
        factors: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        portfolio: Option<PathBuf>,
        #[arg(long)]
        holdings: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        include_weekends: bool,
    },
}

#[derive(Args, Clone)]
struct AnalyzeArgs {
    #[arg(
        long,
        conflicts_with_all = ["postgres_url", "query", "query_file"]
    )]
    input: Option<PathBuf>,
    #[arg(long)]
    postgres_url: Option<String>,
    #[arg(long, value_enum, default_value = "prefer")]
    postgres_ssl_mode: PostgresSslMode,
    #[arg(long)]
    postgres_ca_file: Option<PathBuf>,
    #[arg(long, conflicts_with = "query_file")]
    query: Option<String>,
    #[arg(long, conflicts_with = "query")]
    query_file: Option<PathBuf>,
    #[arg(long)]
    profile: Option<String>,
    #[arg(long)]
    profile_config: Option<PathBuf>,
    #[arg(long, value_delimiter = ',')]
    group_by: Vec<String>,
    #[arg(long, default_value_t = 5)]
    auto_group_k: usize,
    #[arg(long, value_delimiter = ',')]
    metrics: Vec<String>,
    #[arg(long, default_value_t = false)]
    count_only: bool,
    #[arg(long, value_enum, default_value = "sum")]
    agg: AggKind,
    #[arg(long, value_delimiter = ',')]
    percentiles: Vec<PercentileKind>,
    #[arg(long, default_value_t = false)]
    normalize_text_groups: bool,
    #[arg(long, default_value_t = false)]
    word_freq: bool,
    #[arg(long)]
    date_column: Option<String>,
    #[arg(long, value_enum)]
    time_grain: Option<TimeGrain>,
    #[arg(long, value_enum)]
    period: Option<PeriodPreset>,
    #[arg(long)]
    anchor_date: Option<String>,
    #[arg(long)]
    current_start: Option<String>,
    #[arg(long)]
    current_end: Option<String>,
    #[arg(long)]
    previous_start: Option<String>,
    #[arg(long)]
    previous_end: Option<String>,
    #[arg(long, value_delimiter = ',')]
    r#where: Vec<String>,
    #[arg(long, default_value_t = false)]
    exclude_blank_groups: bool,
    #[arg(long)]
    rank_by: Option<String>,
    #[arg(long, default_value_t = 20)]
    top: usize,
    #[arg(long, default_value_t = 0)]
    top_insights: usize,
    #[arg(long, default_value_t = 2)]
    opportunity_min_records: u64,
    #[arg(long, default_value_t = 1)]
    min_records: u64,
    #[arg(long)]
    alert_top5_share: Option<f64>,
    #[arg(long)]
    alert_blank_share: Option<f64>,
    #[arg(long, value_delimiter = ',')]
    alert_rule: Vec<String>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "both")]
    output_format: OutputFormat,
}

#[derive(Args, Clone)]
struct AnalyzeCompareArgs {
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    new: PathBuf,
    #[arg(long, default_value_t = 10)]
    top_movers: usize,
    #[arg(long, value_enum, default_value = "both")]
    output_format: CompareOutputFormat,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args, Clone)]
struct AnalyzeSuggestArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "suggested")]
    profile_name: String,
    #[arg(long, default_value_t = 3)]
    auto_group_k: usize,
    #[arg(long, default_value_t = 3)]
    max_metrics: usize,
    #[arg(long, default_value_t = 2000)]
    sample_rows: usize,
    #[arg(long, value_enum, default_value = "head")]
    sample_mode: SampleMode,
    #[arg(long, default_value_t = 42)]
    sample_seed: u64,
    #[arg(long)]
    out_profile: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "both")]
    output_format: SuggestOutputFormat,
}

#[derive(Args, Clone)]
struct InvestigateArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    metric: String,
    #[arg(long, value_delimiter = ',')]
    drivers: Vec<String>,
    #[arg(long, value_enum)]
    driver_preset: Option<DriverPreset>,
    #[arg(long, value_enum, default_value = "deterministic")]
    auto_drivers: AutoDriversMode,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    dedup_drivers: bool,
    #[arg(long, value_enum, default_value = "percent")]
    driver_contrib: InvestigateContribMode,
    #[arg(long, default_value_t = 3)]
    top_drivers: usize,
    #[arg(long, value_enum, default_value = "both")]
    output_format: InvestigateOutputFormat,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = 3)]
    max_id_drivers: usize,
    #[arg(long, default_value_t = 2)]
    max_cat_drivers: usize,
    #[arg(long, default_value_t = 2)]
    max_num_drivers: usize,
    #[arg(long)]
    date_column: Option<String>,
    #[arg(long, value_enum)]
    time_grain: Option<TimeGrain>,
    #[arg(long, value_enum)]
    period: Option<PeriodPreset>,
    #[arg(long)]
    anchor_date: Option<String>,
    #[arg(long)]
    current_start: Option<String>,
    #[arg(long)]
    current_end: Option<String>,
    #[arg(long)]
    previous_start: Option<String>,
    #[arg(long)]
    previous_end: Option<String>,
}

#[derive(Args, Clone)]
struct AnalyzeDriversArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    input_new: Option<PathBuf>,
    #[arg(long)]
    metric: String,
    #[arg(long, value_enum, default_value = "both")]
    output_format: InvestigateOutputFormat,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    date_column: Option<String>,
    #[arg(long, value_enum)]
    time_grain: Option<TimeGrain>,
    #[arg(long, value_enum)]
    period: Option<PeriodPreset>,
    #[arg(long)]
    anchor_date: Option<String>,
    #[arg(long)]
    current_start: Option<String>,
    #[arg(long)]
    current_end: Option<String>,
    #[arg(long)]
    previous_start: Option<String>,
    #[arg(long)]
    previous_end: Option<String>,
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum CompareOutputFormat {
    Md,
    Html,
    Json,
    Both,
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum SuggestOutputFormat {
    Md,
    Json,
    Both,
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum SampleMode {
    Head,
    Random,
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum AutoDriversMode {
    Deterministic,
    NumericCorr,
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum DriverPreset {
    Id,
    Amount,
    Category,
    Mixed,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum InvestigateContribMode {
    Percent,
    Amount,
    Both,
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum InvestigateOutputFormat {
    Md,
    Json,
    Both,
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum BackendArg {
    Local,
    Bedrock,
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum AggKind {
    Sum,
    Mean,
    Median,
}

impl AggKind {
    fn label(self) -> &'static str {
        match self {
            AggKind::Sum => "Sum",
            AggKind::Mean => "Mean",
            AggKind::Median => "Median",
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum PercentileKind {
    P50,
    P90,
}

impl PercentileKind {
    fn label(self) -> &'static str {
        match self {
            PercentileKind::P50 => "p50",
            PercentileKind::P90 => "p90",
        }
    }

    fn quantile(self) -> f64 {
        match self {
            PercentileKind::P50 => 0.50,
            PercentileKind::P90 => 0.90,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Md,
    Json,
    Both,
    Html,
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum PostgresSslMode {
    Disable,
    Prefer,
    Require,
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum, Debug)]
enum TimeGrain {
    Day,
    Week,
    Month,
    Year,
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum, Debug)]
enum PeriodPreset {
    Current,
    Previous,
    Last,
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
                    k_auto,
                    out,
                    portfolio,
                    holdings,
                    include_weekends,
                },
        } => {
            if portfolio.is_some() && holdings.is_some() {
                return Err(anyhow!("use either --portfolio or --holdings, not both"));
            }
            let mut prices = read_prices_csv(&prices)?;
            if !include_weekends {
                prices.retain(|p| !matches!(p.date.weekday(), Weekday::Sat | Weekday::Sun));
            }

            let k_eff = if k_auto {
                let assets = distinct_tickers(&prices).len();
                assets.saturating_sub(1).clamp(1, 10)
            } else {
                k
            };
            let model = fit_pca(&prices, k_eff)?;

            let weights = if let Some(path) = holdings {
                read_holdings_as_weights_csv(&path)?
            } else if let Some(path) = portfolio {
                read_portfolio_csv(&path)?
            } else {
                equal_weights(&model.tickers)
            };

            let portfolio_returns = portfolio_returns_from_weights(&model, &weights)?;
            let contributions = portfolio_factor_contributions(&model, &weights)?;
            let outliers = compute_outliers(&model.dates, &portfolio_returns, &contributions, 15);
            write_factor_artifacts(&out, &model, &contributions, &outliers)?;

            println!("Wrote artifacts to {}", out.display());
            if k_auto {
                println!("Factors k: auto-selected to {}", model.k);
            } else {
                println!("Factors k: {}", model.k);
            }
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
        Commands::Factors {
            command:
                FactorsCommand::Regress {
                    prices,
                    factors,
                    out,
                    portfolio,
                    holdings,
                    include_weekends,
                },
        } => {
            if portfolio.is_some() && holdings.is_some() {
                return Err(anyhow!("use either --portfolio or --holdings, not both"));
            }

            let mut prices = read_prices_csv(&prices)?;
            if !include_weekends {
                prices.retain(|p| !matches!(p.date.weekday(), Weekday::Sat | Weekday::Sun));
            }
            let factors_df = read_factors_csv(&factors)?;

            let tickers = distinct_tickers(&prices);
            let weights = if let Some(path) = holdings {
                read_holdings_as_weights_csv(&path)?
            } else if let Some(path) = portfolio {
                read_portfolio_csv(&path)?
            } else {
                equal_weights(&tickers)
            };

            let aligned = align_portfolio_and_factors(&prices, &weights, &factors_df)?;
            let reg = ols_regression(
                &aligned.dates,
                &aligned.y,
                &aligned.x,
                &aligned.factor_names,
            )?;
            write_regression_artifacts(&out, &reg)?;

            println!("Wrote regression artifacts to {}", out.display());
            println!("- {}", out.join("regression.json").display());
            println!("- {}", out.join("regression_residuals.csv").display());
            println!(
                "Known-factor regression fit: observations={}, r2={:.4}",
                reg.observations, reg.r2
            );
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
            let mut labels = auto_factor_labels(&summary);
            let custom_labels = load_factor_labels(factor_labels.as_ref())?;
            labels.extend(custom_labels);
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
        Commands::ExplainAnalyze {
            backend,
            model,
            analysis_json,
            question,
        } => {
            let analysis = fs::read_to_string(&analysis_json).map_err(|e| {
                anyhow!(
                    "failed to read analysis json '{}': {}",
                    analysis_json.display(),
                    e
                )
            })?;
            let v: serde_json::Value = serde_json::from_str(&analysis).map_err(|e| {
                anyhow!(
                    "failed to parse analysis json '{}': {}",
                    analysis_json.display(),
                    e
                )
            })?;

            let backend = match backend {
                BackendArg::Local => Backend::Local,
                BackendArg::Bedrock => Backend::Bedrock,
            };
            let client = build_client(backend, model);
            let evidence = build_analysis_evidence(&v);
            let context = build_analysis_prompt_context(&v, &evidence);
            let system = "You are an analytics assistant. Use only provided analysis context. If missing, say unknown. Respond in plain text with concise bullets and concrete actions. Cite evidence IDs like [E1], [E2] for each claim.";
            let user = format!("Question: {}\n\nAnalysis context:\n{}", question, context);
            let answer = client.answer(system, &user)?;
            println!("{}", answer.trim());
            if !evidence.is_empty() {
                println!("\nEvidence:");
                for (i, line) in evidence.iter().enumerate() {
                    println!("- [E{}] {}", i + 1, line);
                }
            }
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
        Commands::Analyze(args) => {
            run_analyze(args)?;
        }
        Commands::AnalyzePeriod(args) => {
            run_analyze(args)?;
        }
        Commands::AnalyzeValidate(args) => {
            run_analyze_validate(args)?;
        }
        Commands::AnalyzeInvestigate(args) => {
            run_analyze_investigate(args)?;
        }
        Commands::AnalyzeDrivers(args) => {
            run_analyze_drivers(args)?;
        }
        Commands::AnalyzeSuggest(args) => {
            run_analyze_suggest(args)?;
        }
        Commands::AnalyzeCompare(args) => {
            run_analyze_compare(args)?;
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

fn run_analyze(args: AnalyzeArgs) -> Result<()> {
    let args = apply_analyze_profile(args)?;
    let period_cfg = parse_period_compare_config(&args)?;
    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| default_analyze_out(&args));
    ensure_parent_dir(&out_path)?;
    let (input_path, _temp_path) = materialize_analyze_input(&args)?;
    let report = analyze_table_csv(
        &input_path,
        args.profile.as_deref(),
        &args.group_by,
        args.auto_group_k,
        &args.metrics,
        args.count_only,
        args.agg,
        &args.percentiles,
        args.normalize_text_groups,
        args.word_freq,
        period_cfg.as_ref(),
        &args.r#where,
        args.exclude_blank_groups,
        args.rank_by.as_deref(),
        args.top,
        args.top_insights,
        args.opportunity_min_records,
        args.min_records,
        args.alert_top5_share,
        args.alert_blank_share,
        &args.alert_rule,
    )?;
    match args.output_format {
        OutputFormat::Md => {
            fs::write(&out_path, report.markdown)?;
            println!("Analysis (markdown) written to {}", out_path.display());
        }
        OutputFormat::Json => {
            fs::write(&out_path, serde_json::to_string_pretty(&report.json)?)?;
            println!("Analysis (json) written to {}", out_path.display());
        }
        OutputFormat::Both => {
            fs::write(&out_path, report.markdown)?;
            let json_path = out_path.with_extension("json");
            fs::write(&json_path, serde_json::to_string_pretty(&report.json)?)?;
            println!("Analysis written to {}", out_path.display());
            println!("Analysis JSON written to {}", json_path.display());
        }
        OutputFormat::Html => {
            let html = markdown_to_html(&report.markdown);
            fs::write(&out_path, html)?;
            println!("Analysis (html) written to {}", out_path.display());
        }
    }
    println!(
        "Detected/used groups: {}",
        report
            .used_groups
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

fn run_analyze_validate(args: AnalyzeArgs) -> Result<()> {
    let args = apply_analyze_profile(args)?;
    let period_cfg = parse_period_compare_config(&args)?;
    let (input_path, _temp_path) = materialize_analyze_input(&args)?;
    let mut rdr = csv::Reader::from_path(&input_path)?;
    let headers = rdr.headers()?.clone();

    let resolved_groups = if args.group_by.is_empty() {
        auto_detect_groups(&headers, &input_path, args.auto_group_k)?
    } else {
        args.group_by
            .iter()
            .map(|g| resolve_group_name(g, &headers))
            .collect::<Result<Vec<_>>>()?
    };
    if resolved_groups.is_empty() {
        return Err(anyhow!("no grouping columns selected or detected"));
    }

    let metric_cols = if args.count_only {
        Vec::new()
    } else if args.metrics.is_empty() {
        auto_detect_numeric_metrics(&input_path, &headers, &resolved_groups, 3)?
    } else {
        args.metrics
            .iter()
            .map(|m| {
                headers
                    .iter()
                    .position(|h| h == m)
                    .map(|idx| (m.to_string(), idx))
                    .ok_or_else(|| anyhow!("metric column '{}' not found", m))
            })
            .collect::<Result<Vec<_>>>()?
    };

    if args.count_only {
        if let Some(rb) = args.rank_by.as_deref() {
            if rb != "count" {
                return Err(anyhow!(
                    "--count-only supports ranking by count only; remove --rank-by or use --rank-by count"
                ));
            }
        }
    } else if let Some(rb) = args.rank_by.as_deref() {
        let rank_metric = metric_cols
            .iter()
            .find(|(m, _)| m == rb)
            .map(|(m, _)| m.clone());
        if rank_metric.is_none() {
            return Err(anyhow!(
                "rank metric '{}' not found. Available metrics: {}",
                rb,
                metric_cols
                    .iter()
                    .map(|(m, _)| m.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }

    let _group_idxs = resolved_groups
        .iter()
        .map(|g| {
            headers
                .iter()
                .position(|h| h == g)
                .ok_or_else(|| anyhow!("group column not found: {}", g))
        })
        .collect::<Result<Vec<_>>>()?;
    let where_filters = parse_where_filters(&args.r#where, &headers)?;
    for raw in &args.alert_rule {
        let _ = parse_alert_rule(raw)?;
    }
    if let Some(cfg) = &period_cfg {
        let _ = resolve_group_name(&cfg.date_column, &headers)?;
    }

    println!("Analyze validation: OK");
    println!("- Input: {}", input_path.display());
    println!("- Profile: {}", args.profile.as_deref().unwrap_or("(none)"));
    println!("- Groups: {}", resolved_groups.join(", "));
    if args.count_only {
        println!("- Metrics: disabled via --count-only");
    } else if metric_cols.is_empty() {
        println!("- Metrics: none detected");
    } else {
        println!(
            "- Metrics: {}",
            metric_cols
                .iter()
                .map(|(m, _)| m.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!(
        "- Rank by: {}",
        args.rank_by.as_deref().unwrap_or("count/default")
    );
    println!("- Aggregation: {}", args.agg.label());
    if args.percentiles.is_empty() {
        println!("- Percentiles: none");
    } else {
        println!(
            "- Percentiles: {}",
            args.percentiles
                .iter()
                .map(|p| p.label())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!(
        "- Filters: {}",
        if where_filters.is_empty() {
            "none".to_string()
        } else {
            where_filters
                .iter()
                .map(|(c, _, v)| format!("{}={}", c, v))
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!("- Alert rules: {}", args.alert_rule.len());
    if let Some(cfg) = &period_cfg {
        println!(
            "- Period compare: enabled ({}, current={}..{}, previous={}..{})",
            cfg.date_column,
            cfg.current_start,
            cfg.current_end,
            cfg.previous_start,
            cfg.previous_end
        );
    } else {
        println!("- Period compare: disabled");
    }
    println!("- Headers detected: {}", headers.len());
    Ok(())
}

fn run_analyze_investigate(args: InvestigateArgs) -> Result<()> {
    if args.top_drivers < 1 {
        return Err(anyhow!("--top-drivers must be >= 1"));
    }
    if !args.drivers.is_empty() && args.driver_preset.is_some() {
        return Err(anyhow!(
            "use either --drivers or --driver-preset, not both"
        ));
    }
    let mut rdr = csv::Reader::from_path(&args.input)?;
    let headers = rdr.headers()?.clone();
    let metric_name = resolve_group_name(&args.metric, &headers)?;
    let metric_idx = headers
        .iter()
        .position(|h| h == metric_name)
        .ok_or_else(|| anyhow!("metric '{}' not found", metric_name))?;

    let mut rows = Vec::<StringRecord>::new();
    for rec in rdr.records() {
        rows.push(rec?);
    }
    if rows.is_empty() {
        return Err(anyhow!("input CSV has no rows"));
    }

    let date_col_name = if let Some(dc) = &args.date_column {
        Some(resolve_group_name(dc, &headers)?)
    } else {
        auto_detect_date_column(&headers, &rows)
    };
    let date_idx = date_col_name.as_ref().and_then(|n| headers.iter().position(|h| h == n));
    let didx = date_idx.ok_or_else(|| {
        anyhow!(
            "date column is required for investigate mode; pass --date-column or provide a detectable date column"
        )
    })?;

    let period_cfg = parse_period_cfg_from_investigate(&args, date_col_name.clone().unwrap_or_default())?;

    let mut curr_metric = 0.0_f64;
    let mut prev_metric = 0.0_f64;
    let mut curr_rows = 0_u64;
    let mut prev_rows = 0_u64;

    let mut driver_specs = if args.drivers.is_empty() {
        if let Some(preset) = args.driver_preset {
            select_driver_specs_by_preset(&args, preset, &headers, &rows, metric_idx, didx)
        } else {
            auto_select_driver_specs(&args, &headers, &rows, metric_idx, didx)
        }
    } else {
        args.drivers
            .iter()
            .map(|d| parse_investigate_driver(d, &headers))
            .collect::<Result<Vec<_>>>()?
    };
    driver_specs = dedup_driver_specs(driver_specs);
    if driver_specs.is_empty() {
        return Err(anyhow!(
            "no usable driver columns found; pass --drivers explicitly"
        ));
    }

    let mut driver_state = driver_specs
        .iter()
        .map(|d| {
            (
                d.label.clone(),
                DriverState {
                    curr_sum: 0.0,
                    prev_sum: 0.0,
                    curr_count: 0,
                    prev_count: 0,
                    curr_distinct: HashSet::new(),
                    prev_distinct: HashSet::new(),
                },
            )
        })
        .collect::<HashMap<_, _>>();

    for rec in &rows {
        let d = parse_date_like(rec.get(didx).unwrap_or("").trim());
        let Some(d) = d else { continue };
        let is_curr = d >= period_cfg.current_start && d <= period_cfg.current_end;
        let is_prev = d >= period_cfg.previous_start && d <= period_cfg.previous_end;
        if !is_curr && !is_prev {
            continue;
        }

        let mv = parse_numeric(rec.get(metric_idx).unwrap_or("").trim()).unwrap_or(0.0);
        if is_curr {
            curr_metric += mv;
            curr_rows += 1;
        }
        if is_prev {
            prev_metric += mv;
            prev_rows += 1;
        }

        for spec in &driver_specs {
            let state = driver_state
                .get_mut(&spec.label)
                .ok_or_else(|| anyhow!("internal error: missing driver state"))?;
            let raw = spec
                .col_idx
                .and_then(|idx| rec.get(idx))
                .unwrap_or("")
                .trim();
            match spec.agg {
                DriverAgg::Sum => {
                    let v = parse_numeric(raw).unwrap_or(0.0);
                    if is_curr {
                        state.curr_sum += v;
                    }
                    if is_prev {
                        state.prev_sum += v;
                    }
                }
                DriverAgg::Mean => {
                    if let Some(v) = parse_numeric(raw) {
                        if is_curr {
                            state.curr_sum += v;
                            state.curr_count += 1;
                        }
                        if is_prev {
                            state.prev_sum += v;
                            state.prev_count += 1;
                        }
                    }
                }
                DriverAgg::Count => {
                    let has_value = spec.col_idx.is_none() || !raw.is_empty();
                    if has_value {
                        if is_curr {
                            state.curr_sum += 1.0;
                        }
                        if is_prev {
                            state.prev_sum += 1.0;
                        }
                    }
                }
                DriverAgg::CountDistinct => {
                    if !raw.is_empty() {
                        if is_curr {
                            state.curr_distinct.insert(raw.to_string());
                        }
                        if is_prev {
                            state.prev_distinct.insert(raw.to_string());
                        }
                    }
                }
            }
        }
    }

    if curr_rows == 0 || prev_rows == 0 {
        return Err(anyhow!(
            "period windows contain no comparable rows (current={}, previous={})",
            curr_rows,
            prev_rows
        ));
    }

    let metric_change_pct = if prev_metric.abs() > 1e-12 {
        ((curr_metric - prev_metric) / prev_metric.abs()) * 100.0
    } else {
        0.0
    };

    let mut driver_curr = HashMap::<String, f64>::new();
    let mut driver_prev = HashMap::<String, f64>::new();
    let mut driver_delta = HashMap::<String, f64>::new();
    for spec in &driver_specs {
        let state = driver_state
            .get(&spec.label)
            .ok_or_else(|| anyhow!("internal error: missing finalized driver state"))?;
        let (cv, pv) = match spec.agg {
            DriverAgg::Sum | DriverAgg::Count => (state.curr_sum, state.prev_sum),
            DriverAgg::Mean => (
                if state.curr_count > 0 {
                    state.curr_sum / state.curr_count as f64
                } else {
                    0.0
                },
                if state.prev_count > 0 {
                    state.prev_sum / state.prev_count as f64
                } else {
                    0.0
                },
            ),
            DriverAgg::CountDistinct => (
                state.curr_distinct.len() as f64,
                state.prev_distinct.len() as f64,
            ),
        };
        driver_curr.insert(spec.label.clone(), cv);
        driver_prev.insert(spec.label.clone(), pv);
        driver_delta.insert(spec.label.clone(), cv - pv);
    }

    let residual_model = investigate_residual_summary(
        &headers,
        &rows,
        metric_idx,
        didx,
        &period_cfg,
        &driver_specs,
        curr_metric,
        prev_metric,
    );
    let mut components = Vec::<(String, f64)>::new();
    if !residual_model.driver_contributions.is_empty() {
        components = residual_model
            .driver_contributions
            .iter()
            .map(|(name, (pct, _amount))| (name.clone(), *pct))
            .collect::<Vec<_>>();
    } else {
        let mut log_terms = Vec::<(String, f64)>::new();
        for spec in &driver_specs {
            let name = &spec.label;
            let cv = *driver_curr.get(name).unwrap_or(&0.0);
            let pv = *driver_prev.get(name).unwrap_or(&0.0);
            if cv > 0.0 && pv > 0.0 {
                log_terms.push((name.clone(), (cv / pv).ln()));
            }
        }

        let denom = log_terms.iter().map(|(_, v)| *v).sum::<f64>();
        if denom.abs() > 1e-12 {
            components = log_terms
                .into_iter()
                .map(|(n, t)| (n, (t / denom) * metric_change_pct))
                .collect::<Vec<_>>();
        } else {
            let mut raw = Vec::<(String, f64)>::new();
            let mut total_abs = 0.0_f64;
            for spec in &driver_specs {
                let name = &spec.label;
                let d = *driver_curr.get(name).unwrap_or(&0.0)
                    - *driver_prev.get(name).unwrap_or(&0.0);
                total_abs += d.abs();
                raw.push((name.clone(), d));
            }
            if total_abs > 1e-12 {
                components = raw
                    .into_iter()
                    .map(|(n, d)| {
                        (
                            n,
                            (d.abs() / total_abs)
                                * metric_change_pct.signum()
                                * d.signum()
                                * metric_change_pct.abs(),
                        )
                    })
                    .collect::<Vec<_>>();
            }
        }
    }

    components.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let agg_by_name = driver_specs
        .iter()
        .map(|s| (s.label.clone(), s.agg))
        .collect::<HashMap<_, _>>();
    let filtered = components
        .iter()
        .filter(|(name, _)| {
            let delta = *driver_delta.get(name).unwrap_or(&0.0);
            let agg = agg_by_name.get(name).copied();
            // Hide near-zero cardinality movements; they are usually not informative.
            !(matches!(agg, Some(DriverAgg::CountDistinct)) && delta.abs() < 2.0)
        })
        .collect::<Vec<_>>();
    let chosen = if filtered.is_empty() {
        components.iter().collect::<Vec<_>>()
    } else {
        filtered
    };

    let top = chosen
        .into_iter()
        .take(args.top_drivers)
        .map(|(name, pct)| {
            let delta = residual_model
                .driver_contributions
                .get(name)
                .map(|(_, amount)| *amount)
                .unwrap_or_else(|| *driver_delta.get(name).unwrap_or(&0.0));
            (name.clone(), *pct, delta)
        })
        .collect::<Vec<_>>();
    let residual = residual_model.summary;
    let explained_pct = top.iter().map(|(_, pct, _)| *pct).sum::<f64>();
    let explained_share_pct = if metric_change_pct.abs() > 1e-12 {
        (explained_pct / metric_change_pct) * 100.0
    } else {
        0.0
    };

    println!("{} change: {:+.1}%", metric_name, metric_change_pct);
    println!();
    println!(
        "Window: {}..{} vs {}..{}",
        period_cfg.current_start,
        period_cfg.current_end,
        period_cfg.previous_start,
        period_cfg.previous_end
    );
    println!();
    println!("Decomposition mode: {}", residual_model.decomposition_mode);
    println!();
    println!("Driver contributions");
    for (name, c, d) in &top {
        match args.driver_contrib {
            InvestigateContribMode::Percent => {
                println!("- {}: {:+.1}%", name, c);
            }
            InvestigateContribMode::Amount => {
                println!("- {}: delta={:+.2}", name, d);
            }
            InvestigateContribMode::Both => {
                println!("- {}: {:+.1}% | delta={:+.2}", name, c, d);
            }
        }
    }
    println!();
    println!("Closure check");
    println!("- explained: {:+.1}% ({:.0}%)", explained_pct, explained_share_pct);
    println!(
        "- residual: {:+.1}% ({})",
        residual.residual_pct,
        signed_fmt_num_commas(residual.residual_amount, 2)
    );
    if !residual.signals.is_empty() {
        println!();
        println!("Residual segments");
        for signal in residual.signals.iter().take(3) {
            println!("- {}: {}", pretty_signal_name(&signal.name), signal.detail);
        }
    }

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| default_analyze_investigate_out(&args));
    ensure_parent_dir(&out_path)?;
    let markdown = render_investigate_markdown(
        &args.input,
        &metric_name,
        metric_change_pct,
        curr_metric,
        prev_metric,
        &period_cfg,
        residual_model.decomposition_mode,
        explained_pct,
        explained_share_pct,
        &top,
        &residual,
    );
    let json = serde_json::json!({
        "input": args.input.display().to_string(),
        "metric": metric_name,
        "metric_change_pct": metric_change_pct,
        "current_metric": curr_metric,
        "previous_metric": prev_metric,
        "window": {
            "current_start": period_cfg.current_start.to_string(),
            "current_end": period_cfg.current_end.to_string(),
            "previous_start": period_cfg.previous_start.to_string(),
            "previous_end": period_cfg.previous_end.to_string(),
            "date_column": period_cfg.date_column,
        },
        "drivers": top.iter().map(|(name, pct, delta)| {
            serde_json::json!({
                "name": name,
                "contrib_pct": pct,
                "delta": delta,
            })
        }).collect::<Vec<_>>(),
        "residual": residual,
        "contribution_view": format!("{:?}", args.driver_contrib).to_lowercase(),
        "decomposition_mode": residual_model.decomposition_mode,
    });
    match args.output_format {
        InvestigateOutputFormat::Md => {
            fs::write(&out_path, markdown)?;
            println!("Investigate (markdown) written to {}", out_path.display());
        }
        InvestigateOutputFormat::Json => {
            fs::write(&out_path, serde_json::to_string_pretty(&json)?)?;
            println!("Investigate (json) written to {}", out_path.display());
        }
        InvestigateOutputFormat::Both => {
            let (md_path, json_path) = investigate_both_paths(&out_path);
            ensure_parent_dir(&md_path)?;
            ensure_parent_dir(&json_path)?;
            fs::write(&md_path, markdown)?;
            fs::write(&json_path, serde_json::to_string_pretty(&json)?)?;
            println!("Investigate written to {}", md_path.display());
            println!("Investigate JSON written to {}", json_path.display());
        }
    }

    Ok(())
}

#[derive(Copy, Clone)]
enum DriverIdentityOp {
    Multiply,
    Divide,
}

struct DriverIdentity {
    metric: String,
    left: String,
    right: String,
    left_idx: usize,
    right_idx: usize,
    left_agg: DriverAgg,
    right_agg: DriverAgg,
    op: DriverIdentityOp,
    fit_mape: f64,
    fit_rows: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ResidualSignal {
    name: String,
    signal_type: String,
    score: f64,
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct ResidualSummary {
    residual_pct: f64,
    residual_amount: f64,
    signals: Vec<ResidualSignal>,
}

struct InvestigateResidualModel {
    decomposition_mode: &'static str,
    summary: ResidualSummary,
    driver_contributions: HashMap<String, (f64, f64)>,
}

fn run_analyze_drivers(args: AnalyzeDriversArgs) -> Result<()> {
    let (headers, base_rows, new_rows) = load_driver_compare_frames(&args)?;
    if base_rows.is_empty() || new_rows.is_empty() {
        return Err(anyhow!(
            "analyze-drivers needs non-empty base and new frames"
        ));
    }

    let metric_name = resolve_group_name(&args.metric, &headers)?;
    let metric_idx = headers
        .iter()
        .position(|h| h == metric_name)
        .ok_or_else(|| anyhow!("metric '{}' not found", metric_name))?;

    let identity = infer_driver_identity(&headers, &base_rows, &new_rows, metric_idx)?
        .ok_or_else(|| anyhow!("no stable 2-term identity found for '{}'", metric_name))?;

    let metric_base = aggregate_column(&base_rows, metric_idx, DriverAgg::Sum);
    let metric_new = aggregate_column(&new_rows, metric_idx, DriverAgg::Sum);
    let total_change_pct = pct_change(metric_new, metric_base);

    let left_base = aggregate_column(&base_rows, identity.left_idx, identity.left_agg);
    let left_new = aggregate_column(&new_rows, identity.left_idx, identity.left_agg);
    let right_base = aggregate_column(&base_rows, identity.right_idx, identity.right_agg);
    let right_new = aggregate_column(&new_rows, identity.right_idx, identity.right_agg);

    // For multiplicative/divisive identities, split the modeled change by each
    // side's log-ratio contribution so the two driver percentages add back up.
    let left_term = safe_ln_ratio(left_new, left_base);
    let right_term_raw = safe_ln_ratio(right_new, right_base);
    let right_term = match identity.op {
        DriverIdentityOp::Multiply => right_term_raw,
        DriverIdentityOp::Divide => -right_term_raw,
    };
    let denom = left_term + right_term;
    // Use row-level predicted metric totals from the inferred identity for
    // closure, rather than mixing in a separate aggregate heuristic.
    let modeled_base = aggregate_identity_prediction(&base_rows, &identity);
    let modeled_new = aggregate_identity_prediction(&new_rows, &identity);
    let explained_change_pct = pct_change(modeled_new, modeled_base);
    let (left_contrib, right_contrib) = if denom.abs() > 1e-12 {
        (
            (left_term / denom) * explained_change_pct,
            (right_term / denom) * explained_change_pct,
        )
    } else {
        (0.0, 0.0)
    };
    // Residual is whatever actual period-over-period movement is left after the
    // inferred identity's modeled change is accounted for.
    let residual_pct = total_change_pct - explained_change_pct;
    let metric_delta = metric_new - metric_base;
    let residual_amount = metric_delta - (modeled_new - modeled_base);
    let residual = analyze_drivers_residual_summary(
        &headers,
        &base_rows,
        &new_rows,
        metric_idx,
        &identity,
        residual_pct,
        residual_amount,
    );

    let period_summary = if args.input_new.is_some() {
        format!(
            "base={} new={}",
            args.input.display(),
            args.input_new.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
        )
    } else {
        let cfg = parse_period_cfg_from_driver_args(&args, headers.clone())?;
        format!(
            "{}..{} vs {}..{}",
            cfg.current_start, cfg.current_end, cfg.previous_start, cfg.previous_end
        )
    };

    let identity_text = match identity.op {
        DriverIdentityOp::Multiply => format!("{} ≈ {} * {}", identity.metric, identity.left, identity.right),
        DriverIdentityOp::Divide => format!("{} ≈ {} / {}", identity.metric, identity.left, identity.right),
    };
    let explained_share_pct = if total_change_pct.abs() > 1e-12 {
        (explained_change_pct / total_change_pct) * 100.0
    } else {
        0.0
    };

    println!("{} change: {:+.1}%", identity.metric, total_change_pct);
    println!();
    println!("Window: {}", period_summary);
    println!();
    println!("Inferred identity");
    println!("- {}", identity_text);
    println!(
        "- fit MAPE: {:.2}% across {} rows",
        identity.fit_mape * 100.0,
        identity.fit_rows
    );
    println!();
    println!("Driver contributions");
    println!("- {}: {:+.1}%", driver_identity_name(&identity.left), left_contrib);
    println!("- {}: {:+.1}%", driver_identity_name(&identity.right), right_contrib);
    println!();
    println!("Closure check");
    println!(
        "- explained: {:+.1}% ({:.0}%)",
        explained_change_pct, explained_share_pct
    );
    println!(
        "- residual: {:+.1}% ({})",
        residual.residual_pct,
        signed_fmt_num_commas(residual.residual_amount, 2)
    );
    if !residual.signals.is_empty() {
        println!();
        println!("Residual segments");
        for signal in residual.signals.iter().take(3) {
            println!("- {}: {}", pretty_signal_name(&signal.name), signal.detail);
        }
    }

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| default_analyze_drivers_out(&args));
    ensure_parent_dir(&out_path)?;
    let markdown = render_analyze_drivers_markdown(
        &args,
        &identity,
        metric_base,
        metric_new,
        total_change_pct,
        explained_change_pct,
        left_base,
        left_new,
        left_contrib,
        right_base,
        right_new,
        right_contrib,
        &residual,
        &period_summary,
    );
    let json = serde_json::json!({
        "input": args.input.display().to_string(),
        "input_new": args.input_new.as_ref().map(|p| p.display().to_string()),
        "metric": identity.metric,
        "identity": {
            "expression": identity_text,
            "left": identity.left,
            "right": identity.right,
            "op": match identity.op {
                DriverIdentityOp::Multiply => "multiply",
                DriverIdentityOp::Divide => "divide",
            },
            "fit_mape": identity.fit_mape,
            "fit_rows": identity.fit_rows,
        },
        "metric_base": metric_base,
        "metric_new": metric_new,
        "metric_change_pct": total_change_pct,
        "drivers": [
            {
                "name": driver_identity_name(&identity.left),
                "base": left_base,
                "new": left_new,
                "contrib_pct": left_contrib
            },
            {
                "name": driver_identity_name(&identity.right),
                "base": right_base,
                "new": right_new,
                "contrib_pct": right_contrib
            }
        ],
        "residual": residual,
        "window": period_summary,
    });

    match args.output_format {
        InvestigateOutputFormat::Md => {
            fs::write(&out_path, markdown)?;
            println!("Analyze drivers (markdown) written to {}", out_path.display());
        }
        InvestigateOutputFormat::Json => {
            fs::write(&out_path, serde_json::to_string_pretty(&json)?)?;
            println!("Analyze drivers (json) written to {}", out_path.display());
        }
        InvestigateOutputFormat::Both => {
            let (md_path, json_path) = investigate_both_paths(&out_path);
            ensure_parent_dir(&md_path)?;
            ensure_parent_dir(&json_path)?;
            fs::write(&md_path, markdown)?;
            fs::write(&json_path, serde_json::to_string_pretty(&json)?)?;
            println!();
            println!("Artifacts written");
            println!("- {}", md_path.display());
            println!("- {}", json_path.display());
        }
    }

    Ok(())
}

fn load_driver_compare_frames(
    args: &AnalyzeDriversArgs,
) -> Result<(StringRecord, Vec<StringRecord>, Vec<StringRecord>)> {
    let mut rdr = csv::Reader::from_path(&args.input)?;
    let headers = rdr.headers()?.clone();
    let mut rows = Vec::<StringRecord>::new();
    for rec in rdr.records() {
        rows.push(rec?);
    }

    if let Some(input_new) = &args.input_new {
        let mut rdr_new = csv::Reader::from_path(input_new)?;
        let headers_new = rdr_new.headers()?.clone();
        if headers.iter().collect::<Vec<_>>() != headers_new.iter().collect::<Vec<_>>() {
            return Err(anyhow!("input and input-new must have identical headers"));
        }
        let mut rows_new = Vec::<StringRecord>::new();
        for rec in rdr_new.records() {
            rows_new.push(rec?);
        }
        return Ok((headers, rows, rows_new));
    }

    let cfg = parse_period_cfg_from_driver_args(args, headers.clone())?;
    let date_name = resolve_group_name(&cfg.date_column, &headers)?;
    let date_idx = headers
        .iter()
        .position(|h| h == date_name)
        .ok_or_else(|| anyhow!("date column '{}' not found", cfg.date_column))?;
    let mut prev_rows = Vec::<StringRecord>::new();
    let mut curr_rows = Vec::<StringRecord>::new();
    for rec in rows {
        let Some(d) = parse_date_like(rec.get(date_idx).unwrap_or("").trim()) else {
            continue;
        };
        if d >= cfg.previous_start && d <= cfg.previous_end {
            prev_rows.push(rec.clone());
        } else if d >= cfg.current_start && d <= cfg.current_end {
            curr_rows.push(rec.clone());
        }
    }
    Ok((headers, prev_rows, curr_rows))
}

fn parse_period_cfg_from_driver_args(
    args: &AnalyzeDriversArgs,
    headers: StringRecord,
) -> Result<PeriodCompareConfig> {
    let date_column = if let Some(dc) = &args.date_column {
        resolve_group_name(dc, &headers)?.to_string()
    } else {
        return Err(anyhow!("--date-column is required when --input-new is not provided"));
    };
    let temp = InvestigateArgs {
        input: args.input.clone(),
        metric: args.metric.clone(),
        drivers: vec![],
        driver_preset: None,
        auto_drivers: AutoDriversMode::Deterministic,
        dedup_drivers: true,
        driver_contrib: InvestigateContribMode::Percent,
        top_drivers: 3,
        output_format: InvestigateOutputFormat::Both,
        out: None,
        max_id_drivers: 3,
        max_cat_drivers: 2,
        max_num_drivers: 2,
        date_column: Some(date_column.clone()),
        time_grain: args.time_grain,
        period: args.period,
        anchor_date: args.anchor_date.clone(),
        current_start: args.current_start.clone(),
        current_end: args.current_end.clone(),
        previous_start: args.previous_start.clone(),
        previous_end: args.previous_end.clone(),
    };
    parse_period_cfg_from_investigate(&temp, date_column)
}

fn infer_driver_identity(
    headers: &StringRecord,
    base_rows: &[StringRecord],
    new_rows: &[StringRecord],
    metric_idx: usize,
) -> Result<Option<DriverIdentity>> {
    let mut candidates = Vec::<DriverIdentity>::new();
    let all_rows = base_rows.iter().chain(new_rows.iter()).collect::<Vec<_>>();
    let numeric_cols = headers
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| {
            if idx == metric_idx {
                return None;
            }
            let ok = all_rows
                .iter()
                .take(5000)
                .filter_map(|rec| rec.get(idx))
                .filter(|raw| parse_numeric(raw.trim()).is_some())
                .count();
            if ok >= 20 {
                Some((idx, name.to_string()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    for i in 0..numeric_cols.len() {
        for j in (i + 1)..numeric_cols.len() {
            let (left_idx, left_name) = &numeric_cols[i];
            let (right_idx, right_name) = &numeric_cols[j];
            for op in [DriverIdentityOp::Multiply, DriverIdentityOp::Divide] {
                let (mape, rows_used) =
                    candidate_identity_fit(all_rows.iter().copied(), metric_idx, *left_idx, *right_idx, op);
                if rows_used >= 20 && mape.is_finite() {
                    candidates.push(DriverIdentity {
                        metric: headers.get(metric_idx).unwrap_or("metric").to_string(),
                        left: left_name.clone(),
                        right: right_name.clone(),
                        left_idx: *left_idx,
                        right_idx: *right_idx,
                        left_agg: infer_numeric_driver_agg(left_name),
                        right_agg: infer_numeric_driver_agg(right_name),
                        op,
                        fit_mape: mape,
                        fit_rows: rows_used,
                    });
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        a.fit_mape
            .partial_cmp(&b.fit_mape)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(candidates.into_iter().find(|c| c.fit_mape <= 0.15))
}

fn candidate_identity_fit<'a>(
    rows: impl Iterator<Item = &'a StringRecord>,
    metric_idx: usize,
    left_idx: usize,
    right_idx: usize,
    op: DriverIdentityOp,
) -> (f64, usize) {
    let mut total_err = 0.0_f64;
    let mut used = 0usize;
    for rec in rows {
        let Some(metric) = parse_numeric(rec.get(metric_idx).unwrap_or("").trim()) else {
            continue;
        };
        let Some(left) = parse_numeric(rec.get(left_idx).unwrap_or("").trim()) else {
            continue;
        };
        let Some(right) = parse_numeric(rec.get(right_idx).unwrap_or("").trim()) else {
            continue;
        };
        if metric.abs() <= 1e-12 {
            continue;
        }
        let pred = match op {
            DriverIdentityOp::Multiply => left * right,
            DriverIdentityOp::Divide => {
                if right.abs() <= 1e-12 {
                    continue;
                }
                left / right
            }
        };
        let rel_err = (pred - metric).abs() / metric.abs().max(1e-12);
        total_err += rel_err;
        used += 1;
    }
    if used == 0 {
        (f64::INFINITY, 0)
    } else {
        (total_err / used as f64, used)
    }
}

fn aggregate_column(rows: &[StringRecord], idx: usize, agg: DriverAgg) -> f64 {
    match agg {
        DriverAgg::Sum | DriverAgg::Count => rows
            .iter()
            .filter_map(|rec| parse_numeric(rec.get(idx).unwrap_or("").trim()))
            .sum::<f64>(),
        DriverAgg::Mean => {
            let vals = rows
                .iter()
                .filter_map(|rec| parse_numeric(rec.get(idx).unwrap_or("").trim()))
                .collect::<Vec<_>>();
            if vals.is_empty() {
                0.0
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            }
        }
        DriverAgg::CountDistinct => {
            let vals = rows
                .iter()
                .map(|rec| rec.get(idx).unwrap_or("").trim())
                .filter(|v| !v.is_empty())
                .collect::<HashSet<_>>();
            vals.len() as f64
        }
    }
}

fn aggregate_identity_prediction(rows: &[StringRecord], identity: &DriverIdentity) -> f64 {
    rows.iter()
        .filter_map(|rec| {
            let left = parse_numeric(rec.get(identity.left_idx).unwrap_or("").trim())?;
            let right = parse_numeric(rec.get(identity.right_idx).unwrap_or("").trim())?;
            match identity.op {
                DriverIdentityOp::Multiply => Some(left * right),
                DriverIdentityOp::Divide => {
                    if right.abs() <= 1e-12 {
                        None
                    } else {
                        Some(left / right)
                    }
                }
            }
        })
        .sum::<f64>()
}

fn pct_change(new_val: f64, base_val: f64) -> f64 {
    if base_val.abs() <= 1e-12 {
        0.0
    } else {
        ((new_val - base_val) / base_val.abs()) * 100.0
    }
}

fn safe_ln_ratio(new_val: f64, base_val: f64) -> f64 {
    if new_val > 0.0 && base_val > 0.0 {
        (new_val / base_val).ln()
    } else {
        0.0
    }
}

fn driver_identity_name(name: &str) -> String {
    name.to_string()
}

fn analyze_drivers_residual_summary(
    headers: &StringRecord,
    base_rows: &[StringRecord],
    new_rows: &[StringRecord],
    metric_idx: usize,
    identity: &DriverIdentity,
    residual_pct: f64,
    residual_amount: f64,
) -> ResidualSummary {
    if residual_pct.abs() < 0.5 && residual_amount.abs() < 10_000.0 {
        return ResidualSummary {
            residual_pct,
            residual_amount,
            signals: Vec::new(),
        };
    }
    let row_residuals = base_rows
        .iter()
        .chain(new_rows.iter())
        .filter_map(|rec| {
            let metric = parse_numeric(rec.get(metric_idx).unwrap_or("").trim())?;
            let left = parse_numeric(rec.get(identity.left_idx).unwrap_or("").trim())?;
            let right = parse_numeric(rec.get(identity.right_idx).unwrap_or("").trim())?;
            let predicted = match identity.op {
                DriverIdentityOp::Multiply => left * right,
                DriverIdentityOp::Divide => {
                    if right.abs() <= 1e-12 {
                        return None;
                    }
                    left / right
                }
            };
            Some((rec, metric - predicted))
        })
        .collect::<Vec<_>>();
    let excluded = HashSet::from([metric_idx, identity.left_idx, identity.right_idx]);
    let signals = residual_signals_from_rows(headers, &row_residuals, &excluded, 3);
    ResidualSummary {
        residual_pct,
        residual_amount,
        signals,
    }
}

fn investigate_residual_summary(
    headers: &StringRecord,
    rows: &[StringRecord],
    metric_idx: usize,
    date_idx: usize,
    period_cfg: &PeriodCompareConfig,
    driver_specs: &[DriverSpec],
    curr_metric: f64,
    prev_metric: f64,
) -> InvestigateResidualModel {
    let numeric_specs = driver_specs
        .iter()
        .filter_map(|spec| match (spec.agg, spec.col_idx) {
            (DriverAgg::Sum | DriverAgg::Mean, Some(idx)) => Some((idx, spec.agg)),
            _ => None,
        })
        .collect::<Vec<_>>();
    if numeric_specs.is_empty() {
        return InvestigateResidualModel {
            decomposition_mode: "heuristic",
            summary: ResidualSummary {
                residual_pct: 0.0,
                residual_amount: 0.0,
                signals: Vec::new(),
            },
            driver_contributions: HashMap::new(),
        };
    }

    let mut row_cache = Vec::<(&StringRecord, f64, Vec<f64>, bool)>::new();

    for rec in rows {
        let Some(d) = parse_date_like(rec.get(date_idx).unwrap_or("").trim()) else {
            continue;
        };
        let is_curr = d >= period_cfg.current_start && d <= period_cfg.current_end;
        let is_prev = d >= period_cfg.previous_start && d <= period_cfg.previous_end;
        if !is_curr && !is_prev {
            continue;
        }
        let Some(metric) = parse_numeric(rec.get(metric_idx).unwrap_or("").trim()) else {
            continue;
        };
        let mut feats = Vec::with_capacity(numeric_specs.len());
        for (idx, _agg) in &numeric_specs {
            if let Some(v) = parse_numeric(rec.get(*idx).unwrap_or("").trim()) {
                feats.push(v);
            } else {
                feats.push(0.0);
            }
        }
        row_cache.push((rec, metric, feats, is_curr));
    }

    if row_cache.len() < 20 {
        return InvestigateResidualModel {
            decomposition_mode: "heuristic",
            summary: ResidualSummary {
                residual_pct: 0.0,
                residual_amount: 0.0,
                signals: Vec::new(),
            },
            driver_contributions: HashMap::new(),
        };
    }

    let p = numeric_specs.len();
    let mut xtx = DMatrix::<f64>::zeros(p + 1, p + 1);
    let mut xty = DVector::<f64>::zeros(p + 1);
    for (_, metric, feats, _) in &row_cache {
        let mut x = Vec::with_capacity(p + 1);
        x.push(1.0);
        x.extend(feats.iter().copied());
        for i in 0..x.len() {
            xty[i] += x[i] * metric;
            for j in 0..x.len() {
                xtx[(i, j)] += x[i] * x[j];
            }
        }
    }
    let Some(beta) = xtx.lu().solve(&xty) else {
        return InvestigateResidualModel {
            decomposition_mode: "heuristic",
            summary: ResidualSummary {
                residual_pct: 0.0,
                residual_amount: 0.0,
                signals: Vec::new(),
            },
            driver_contributions: HashMap::new(),
        };
    };

    let mut predicted_curr = 0.0_f64;
    let mut predicted_prev = 0.0_f64;
    let mut driver_sum_curr = vec![0.0_f64; numeric_specs.len()];
    let mut driver_sum_prev = vec![0.0_f64; numeric_specs.len()];
    let row_residuals = row_cache
        .into_iter()
        .map(|(rec, metric, feats, is_curr)| {
            let mut predicted: f64 = beta[0];
            for (i, feat) in feats.iter().enumerate() {
                predicted += beta[i + 1] * feat;
                if is_curr {
                    driver_sum_curr[i] += *feat;
                } else {
                    driver_sum_prev[i] += *feat;
                }
            }
            if is_curr {
                predicted_curr += predicted;
            } else {
                predicted_prev += predicted;
            }
            (rec, metric - predicted)
        })
        .collect::<Vec<_>>();
    let mut driver_contributions = HashMap::new();
    let mut explained_amount = 0.0_f64;
    for (pos, (idx, agg)) in numeric_specs.iter().enumerate() {
        let delta_amount = beta[pos + 1] * (driver_sum_curr[pos] - driver_sum_prev[pos]);
        let name = match agg {
            DriverAgg::Mean => format!("avg({})", headers.get(*idx).unwrap_or("")),
            _ => format!("sum({})", headers.get(*idx).unwrap_or("")),
        };
        let pct = if prev_metric.abs() > 1e-12 {
            (delta_amount / prev_metric.abs()) * 100.0
        } else {
            0.0
        };
        driver_contributions.insert(name, (pct, delta_amount));
        explained_amount += delta_amount;
    }
    let metric_delta = curr_metric - prev_metric;
    let residual_pct = if prev_metric.abs() > 1e-12 {
        ((metric_delta - explained_amount) / prev_metric.abs()) * 100.0
    } else {
        0.0
    };
    let actual_curr = row_residuals
        .iter()
        .filter(|(rec, _)| {
            let Some(d) = parse_date_like(rec.get(date_idx).unwrap_or("").trim()) else {
                return false;
            };
            d >= period_cfg.current_start && d <= period_cfg.current_end
        })
        .map(|(rec, _)| parse_numeric(rec.get(metric_idx).unwrap_or("").trim()).unwrap_or(0.0))
        .sum::<f64>();
    let actual_prev = row_residuals
        .iter()
        .filter(|(rec, _)| {
            let Some(d) = parse_date_like(rec.get(date_idx).unwrap_or("").trim()) else {
                return false;
            };
            d >= period_cfg.previous_start && d <= period_cfg.previous_end
        })
        .map(|(rec, _)| parse_numeric(rec.get(metric_idx).unwrap_or("").trim()).unwrap_or(0.0))
        .sum::<f64>();
    let residual_amount = (actual_curr - actual_prev) - explained_amount;
    let mut excluded = HashSet::from([metric_idx, date_idx]);
    for spec in driver_specs {
        if let Some(idx) = spec.col_idx {
            excluded.insert(idx);
        }
    }
    let signals = if residual_pct.abs() < 0.5 && residual_amount.abs() < 10_000.0 {
        Vec::new()
    } else {
        residual_signals_from_rows(headers, &row_residuals, &excluded, 3)
    };
    InvestigateResidualModel {
        decomposition_mode: "regression",
        summary: ResidualSummary {
            residual_pct,
            residual_amount,
            signals,
        },
        driver_contributions,
    }
}

fn residual_signals_from_rows(
    headers: &StringRecord,
    row_residuals: &[(&StringRecord, f64)],
    excluded_cols: &HashSet<usize>,
    top_n: usize,
) -> Vec<ResidualSignal> {
    let mut signals = Vec::<ResidualSignal>::new();
    for (idx, name) in headers.iter().enumerate() {
        if excluded_cols.contains(&idx) {
            continue;
        }
        let lname = name.to_ascii_lowercase();
        if lname.contains("date") || lname.contains("time") {
            continue;
        }
        if infer_numeric_column_from_residuals(row_residuals, idx) {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for (rec, residual) in row_residuals {
                if let Some(v) = parse_numeric(rec.get(idx).unwrap_or("").trim()) {
                    xs.push(v);
                    ys.push(*residual);
                }
            }
            if xs.len() >= 20 {
                let corr = pearson_corr(&xs, &ys);
                if corr.is_finite() && corr.abs() >= 0.3 {
                    let direction = if corr >= 0.0 { "higher" } else { "lower" };
                    signals.push(ResidualSignal {
                        name: name.to_string(),
                        signal_type: "numeric".to_string(),
                        score: corr.abs(),
                        detail: format!("{} values correlate with residual (corr={:+.2})", direction, corr),
                    });
                }
            }
        } else {
            let mut by_val: HashMap<String, (f64, usize)> = HashMap::new();
            for (rec, residual) in row_residuals {
                let raw = rec.get(idx).unwrap_or("").trim();
                if raw.is_empty() {
                    continue;
                }
                let entry = by_val.entry(raw.to_string()).or_insert((0.0, 0));
                entry.0 += *residual;
                entry.1 += 1;
            }
            let best = by_val
                .into_iter()
                .filter(|(_, (_, count))| *count >= 3)
                .map(|(value, (sum, count))| {
                    let mean = sum / count as f64;
                    let score = mean.abs() * (count as f64).sqrt();
                    (value, mean, count, score)
                })
                .max_by(|a, b| {
                    a.3.partial_cmp(&b.3)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some((value, mean, count, score)) = best {
                if mean.abs() < 1_000.0 {
                    continue;
                }
                let pretty_name = format!("{} = {}", name, value);
                signals.push(ResidualSignal {
                    name: pretty_name.clone(),
                    signal_type: "categorical".to_string(),
                    score,
                    detail: format!(
                        "mean residual {} ({} rows)",
                        signed_fmt_num_commas(mean, 2),
                        count
                    ),
                });
            }
        }
    }
    signals.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });
    signals.truncate(top_n);
    signals
}

fn infer_numeric_column_from_residuals(row_residuals: &[(&StringRecord, f64)], idx: usize) -> bool {
    let mut parsed = 0usize;
    let mut non_empty = 0usize;
    for (rec, _) in row_residuals.iter().take(5000) {
        let raw = rec.get(idx).unwrap_or("").trim();
        if raw.is_empty() {
            continue;
        }
        non_empty += 1;
        if parse_numeric(raw).is_some() {
            parsed += 1;
        }
    }
    non_empty >= 20 && (parsed as f64 / non_empty as f64) >= 0.8
}

fn render_analyze_drivers_markdown(
    args: &AnalyzeDriversArgs,
    identity: &DriverIdentity,
    _metric_base: f64,
    _metric_new: f64,
    total_change_pct: f64,
    explained_change_pct: f64,
    left_base: f64,
    left_new: f64,
    left_contrib: f64,
    right_base: f64,
    right_new: f64,
    right_contrib: f64,
    residual: &ResidualSummary,
    period_summary: &str,
) -> String {
    let expression = match identity.op {
        DriverIdentityOp::Multiply => format!("{} ≈ {} * {}", identity.metric, identity.left, identity.right),
        DriverIdentityOp::Divide => format!("{} ≈ {} / {}", identity.metric, identity.left, identity.right),
    };
    let explained_share_pct = if total_change_pct.abs() > 1e-12 {
        (explained_change_pct / total_change_pct) * 100.0
    } else {
        0.0
    };
    let mut md = String::new();
    md.push_str("# Driver Decomposition\n\n");
    md.push_str(&format!("- Input: {}\n", args.input.display()));
    if let Some(input_new) = &args.input_new {
        md.push_str(&format!("- Input new: {}\n", input_new.display()));
    }
    md.push_str(&format!("## {} change: {:+.1}%\n\n", identity.metric, total_change_pct));
    md.push_str(&format!("Window: `{}`\n\n", period_summary));
    md.push_str("## Inferred identity\n\n");
    md.push_str(&format!("- `{}`\n", expression));
    md.push_str(&format!(
        "- fit MAPE: {:.2}% across {} rows\n\n",
        identity.fit_mape * 100.0,
        identity.fit_rows
    ));
    md.push_str("## Driver contributions\n\n");
    md.push_str("| Driver | Base | New | Contribution % |\n");
    md.push_str("|---|---:|---:|---:|\n");
    md.push_str(&format!(
        "| {} | {} | {} | {:+.1}% |\n",
        driver_identity_name(&identity.left),
        fmt_num(left_base, 4),
        fmt_num(left_new, 4),
        left_contrib
    ));
    md.push_str(&format!(
        "| {} | {} | {} | {:+.1}% |\n",
        driver_identity_name(&identity.right),
        fmt_num(right_base, 4),
        fmt_num(right_new, 4),
        right_contrib
    ));
    md.push_str("\n## Closure check\n\n");
    md.push_str(&format!(
        "- explained: {:+.1}% ({:.0}%)\n",
        explained_change_pct, explained_share_pct
    ));
    md.push_str(&format!(
        "- residual: {:+.1}% ({}).\n",
        residual.residual_pct,
        signed_fmt_num_commas(residual.residual_amount, 2)
    ));
    if !residual.signals.is_empty() {
        md.push_str("\n## Residual segments\n\n");
        for signal in &residual.signals {
            md.push_str(&format!("- {}: {}\n", pretty_signal_name(&signal.name), signal.detail));
        }
    }
    md.push_str("\n## Artifacts written\n\n");
    let (md_path, json_path) = investigate_both_paths(
        &args.out
            .clone()
            .unwrap_or_else(|| default_analyze_drivers_out(args)),
    );
    md.push_str(&format!("- {}\n", md_path.display()));
    md.push_str(&format!("- {}\n", json_path.display()));
    md
}

fn render_investigate_markdown(
    input: &PathBuf,
    metric: &str,
    metric_change_pct: f64,
    _curr_metric: f64,
    _prev_metric: f64,
    period_cfg: &PeriodCompareConfig,
    decomposition_mode: &str,
    explained_pct: f64,
    explained_share_pct: f64,
    drivers: &[(String, f64, f64)],
    residual: &ResidualSummary,
) -> String {
    let mut md = String::new();
    md.push_str("# Investigate Report\n\n");
    md.push_str(&format!("- Input: {}\n", input.display()));
    md.push_str(&format!("## {} change: {:+.2}%\n\n", metric, metric_change_pct));
    md.push_str(&format!(
        "Window: `{}..{} vs {}..{}`\n\n",
        period_cfg.current_start,
        period_cfg.current_end,
        period_cfg.previous_start,
        period_cfg.previous_end
    ));
    md.push_str(&format!("Decomposition mode: `{}`\n\n", decomposition_mode));
    md.push_str("## Driver contributions\n\n");
    md.push_str("| Driver | Contribution % | Delta |\n");
    md.push_str("|---|---:|---:|\n");
    for (name, pct, delta) in drivers {
        md.push_str(&format!("| {} | {:+.2}% | {:+.4} |\n", name, pct, delta));
    }
    md.push_str("\n## Closure check\n\n");
    md.push_str(&format!("- explained: {:+.2}% ({:.0}%)\n", explained_pct, explained_share_pct));
    md.push_str(&format!(
        "- residual: {:+.2}% ({}).\n",
        residual.residual_pct,
        signed_fmt_num_commas(residual.residual_amount, 2)
    ));
    if !residual.signals.is_empty() {
        md.push_str("\n## Residual segments\n\n");
        for signal in &residual.signals {
            md.push_str(&format!("- {}: {}\n", pretty_signal_name(&signal.name), signal.detail));
        }
    }
    md
}

fn investigate_both_paths(out_path: &PathBuf) -> (PathBuf, PathBuf) {
    let ext = out_path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "json" {
        (out_path.with_extension("md"), out_path.clone())
    } else {
        (out_path.clone(), out_path.with_extension("json"))
    }
}

fn auto_detect_date_column(headers: &StringRecord, rows: &[StringRecord]) -> Option<String> {
    let mut best: Option<(String, f64)> = None;
    for (i, h) in headers.iter().enumerate() {
        let name = h.to_lowercase();
        if !(name.contains("date") || name.contains("time")) {
            continue;
        }
        let mut non_empty = 0usize;
        let mut ok = 0usize;
        for rec in rows.iter().take(1000) {
            let raw = rec.get(i).unwrap_or("").trim();
            if raw.is_empty() {
                continue;
            }
            non_empty += 1;
            if parse_date_like(raw).is_some() {
                ok += 1;
            }
        }
        if non_empty == 0 {
            continue;
        }
        let ratio = ok as f64 / non_empty as f64;
        if ratio >= 0.8 {
            let entry = (h.to_string(), ratio);
            if best.as_ref().map(|(_, r)| ratio > *r).unwrap_or(true) {
                best = Some(entry);
            }
        }
    }
    best.map(|(n, _)| n)
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum DriverAgg {
    Sum,
    Mean,
    Count,
    CountDistinct,
}

struct DriverSpec {
    label: String,
    col_idx: Option<usize>,
    agg: DriverAgg,
}

struct DriverState {
    curr_sum: f64,
    prev_sum: f64,
    curr_count: u64,
    prev_count: u64,
    curr_distinct: HashSet<String>,
    prev_distinct: HashSet<String>,
}

fn dedup_driver_specs(specs: Vec<DriverSpec>) -> Vec<DriverSpec> {
    let mut seen = HashSet::<String>::new();
    let mut out = Vec::<DriverSpec>::new();
    for s in specs {
        if seen.insert(s.label.clone()) {
            out.push(s);
        }
    }
    out
}

fn parse_investigate_driver(raw: &str, headers: &StringRecord) -> Result<DriverSpec> {
    let token = raw.trim();
    if token.is_empty() {
        return Err(anyhow!("empty driver token"));
    }
    if let Some((func, arg)) = parse_driver_fn(token) {
        let agg = match func.as_str() {
            "sum" => DriverAgg::Sum,
            "avg" | "mean" => DriverAgg::Mean,
            "count" => DriverAgg::Count,
            "count_distinct" | "distinct_count" | "n_distinct" => DriverAgg::CountDistinct,
            _ => {
                return Err(anyhow!(
                    "unsupported driver function '{}'; use sum, avg/mean, count, count_distinct",
                    func
                ))
            }
        };

        if agg == DriverAgg::Count && (arg == "*" || arg.is_empty()) {
            return Ok(DriverSpec {
                label: "count(*)".to_string(),
                col_idx: None,
                agg,
            });
        }

        let col_name = resolve_group_name(arg, headers)?;
        let col_idx = headers
            .iter()
            .position(|h| h == col_name)
            .ok_or_else(|| anyhow!("driver column '{}' not found", col_name))?;
        let label = match agg {
            DriverAgg::Sum => format!("sum({})", col_name),
            DriverAgg::Mean => format!("avg({})", col_name),
            DriverAgg::Count => format!("count({})", col_name),
            DriverAgg::CountDistinct => format!("count_distinct({})", col_name),
        };
        return Ok(DriverSpec {
            label,
            col_idx: Some(col_idx),
            agg,
        });
    }

    let col_name = resolve_group_name(token, headers)?;
    let col_idx = headers
        .iter()
        .position(|h| h == col_name)
        .ok_or_else(|| anyhow!("driver '{}' not found", col_name))?;
    Ok(DriverSpec {
        label: col_name,
        col_idx: Some(col_idx),
        agg: DriverAgg::Sum,
    })
}

fn parse_driver_fn(raw: &str) -> Option<(String, &str)> {
    let open = raw.find('(')?;
    if !raw.ends_with(')') || open == 0 {
        return None;
    }
    let fname = raw[..open].trim().to_ascii_lowercase();
    if fname.is_empty() {
        return None;
    }
    let arg = raw[open + 1..raw.len() - 1].trim();
    Some((fname, arg))
}

fn auto_select_driver_specs(
    args: &InvestigateArgs,
    headers: &StringRecord,
    rows: &[StringRecord],
    metric_idx: usize,
    date_idx: usize,
) -> Vec<DriverSpec> {
    match args.auto_drivers {
        AutoDriversMode::NumericCorr => auto_select_numeric_drivers(
            headers,
            rows,
            metric_idx,
            date_idx,
            args.top_drivers.max(1),
            if args.dedup_drivers { Some(0.95) } else { None },
        )
        .into_iter()
        .map(|(n, i)| {
            let agg = infer_numeric_driver_agg(&n);
            DriverSpec {
                label: match agg {
                    DriverAgg::Mean => format!("avg({})", n),
                    _ => format!("sum({})", n),
                },
                col_idx: Some(i),
                agg,
            }
        })
        .collect::<Vec<_>>(),
        AutoDriversMode::Deterministic => {
            let mut specs = Vec::<DriverSpec>::new();
            let mut used_cols = HashSet::<usize>::new();

            let id_name_set = headers
                .iter()
                .map(|h| h.to_ascii_lowercase())
                .collect::<HashSet<_>>();
            let id_drivers = auto_select_id_like_drivers(
                headers,
                metric_idx,
                date_idx,
                args.max_id_drivers,
                &id_name_set,
            );
            for (name, idx) in id_drivers {
                used_cols.insert(idx);
                specs.push(DriverSpec {
                    label: format!("count_distinct({})", name),
                    col_idx: Some(idx),
                    agg: DriverAgg::CountDistinct,
                });
            }

            let cat_drivers = auto_select_categorical_drivers(
                headers,
                rows,
                metric_idx,
                date_idx,
                args.max_cat_drivers,
                &used_cols,
            );
            for (name, idx) in cat_drivers {
                used_cols.insert(idx);
                specs.push(DriverSpec {
                    label: format!("count_distinct({})", name),
                    col_idx: Some(idx),
                    agg: DriverAgg::CountDistinct,
                });
            }

            let num_drivers = auto_select_numeric_drivers(
                headers,
                rows,
                metric_idx,
                date_idx,
                args.max_num_drivers,
                if args.dedup_drivers { Some(0.95) } else { None },
            );
            for (name, idx) in num_drivers {
                if used_cols.contains(&idx) {
                    continue;
                }
                let agg = infer_numeric_driver_agg(&name);
                specs.push(DriverSpec {
                    label: match agg {
                        DriverAgg::Mean => format!("avg({})", name),
                        _ => format!("sum({})", name),
                    },
                    col_idx: Some(idx),
                    agg,
                });
            }
            specs
        }
    }
}

fn select_driver_specs_by_preset(
    args: &InvestigateArgs,
    preset: DriverPreset,
    headers: &StringRecord,
    rows: &[StringRecord],
    metric_idx: usize,
    date_idx: usize,
) -> Vec<DriverSpec> {
    match preset {
        DriverPreset::Id => {
            let mut specs = Vec::<DriverSpec>::new();
            let id_name_set = headers
                .iter()
                .map(|h| h.to_ascii_lowercase())
                .collect::<HashSet<_>>();
            for (name, idx) in auto_select_id_like_drivers(
                headers,
                metric_idx,
                date_idx,
                args.max_id_drivers.max(args.top_drivers),
                &id_name_set,
            ) {
                specs.push(DriverSpec {
                    label: format!("count_distinct({})", name),
                    col_idx: Some(idx),
                    agg: DriverAgg::CountDistinct,
                });
            }
            specs
        }
        DriverPreset::Amount => auto_select_numeric_drivers(
            headers,
            rows,
            metric_idx,
            date_idx,
            args.max_num_drivers.max(args.top_drivers),
            if args.dedup_drivers { Some(0.95) } else { None },
        )
        .into_iter()
        .map(|(name, idx)| {
            let agg = infer_numeric_driver_agg(&name);
            DriverSpec {
                label: match agg {
                    DriverAgg::Mean => format!("avg({})", name),
                    _ => format!("sum({})", name),
                },
                col_idx: Some(idx),
                agg,
            }
        })
        .collect::<Vec<_>>(),
        DriverPreset::Category => auto_select_categorical_drivers(
            headers,
            rows,
            metric_idx,
            date_idx,
            args.max_cat_drivers.max(args.top_drivers),
            &HashSet::new(),
        )
        .into_iter()
        .map(|(name, idx)| DriverSpec {
            label: format!("count_distinct({})", name),
            col_idx: Some(idx),
            agg: DriverAgg::CountDistinct,
        })
        .collect::<Vec<_>>(),
        DriverPreset::Mixed => auto_select_driver_specs(args, headers, rows, metric_idx, date_idx),
    }
}

fn auto_select_id_like_drivers(
    headers: &StringRecord,
    metric_idx: usize,
    date_idx: usize,
    max_n: usize,
    header_name_set: &HashSet<String>,
) -> Vec<(String, usize)> {
    let mut out = headers
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| {
            if idx == metric_idx || idx == date_idx {
                return None;
            }
            let lname = name.to_ascii_lowercase();
            let is_uuid_dup = lname.ends_with("_uuid")
                && header_name_set.contains(&(lname.trim_end_matches("_uuid").to_string() + "_id"));
            if (lname == "id" || lname.ends_with("_id") || lname.ends_with("_uuid")) && !is_uuid_dup {
                Some((name.to_string(), idx))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out.into_iter().take(max_n).collect::<Vec<_>>()
}

fn auto_select_categorical_drivers(
    headers: &StringRecord,
    rows: &[StringRecord],
    metric_idx: usize,
    date_idx: usize,
    max_n: usize,
    used_cols: &HashSet<usize>,
) -> Vec<(String, usize)> {
    let mut scored = Vec::<(String, usize, i32)>::new();
    for (idx, name) in headers.iter().enumerate() {
        if idx == metric_idx || idx == date_idx || used_cols.contains(&idx) {
            continue;
        }
        let lname = name.to_ascii_lowercase();
        if lname.contains("date") || lname.contains("time") {
            continue;
        }
        if infer_numeric_column(rows, idx) {
            continue;
        }
        let (distinct_count, non_empty) = approx_distinct_count(rows, idx, 5000);
        if non_empty < 20 || distinct_count < 2 {
            continue;
        }
        // Skip near-unique text-like columns for category mode.
        let uniq_ratio = distinct_count as f64 / non_empty as f64;
        if uniq_ratio > 0.35 {
            continue;
        }
        let mut score = 0_i32;
        if contains_any(
            &lname,
            &[
                "category",
                "subcategory",
                "discipline",
                "segment",
                "plan",
                "tier",
                "type",
                "channel",
                "region",
                "country",
                "market",
                "vertical",
            ],
        ) {
            score += 30;
        }
        if contains_any(
            &lname,
            &[
                "status",
                "invoice",
                "backoffice",
                "computed",
                "created",
                "updated",
                "last_",
            ],
        ) {
            score -= 30;
        }
        if contains_any(
            &lname,
            &["name", "title", "description", "comment", "note", "email", "address"],
        ) {
            score -= 25;
        }
        // Prefer moderate-cardinality dimensions.
        if distinct_count <= 30 {
            score += 8;
        } else if distinct_count <= 80 {
            score += 4;
        } else {
            score -= 5;
        }
        scored.push((name.to_string(), idx, score));
    }
    scored.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    scored
        .into_iter()
        .take(max_n)
        .map(|(n, i, _)| (n, i))
        .collect::<Vec<_>>()
}

fn infer_numeric_column(rows: &[StringRecord], idx: usize) -> bool {
    let mut parsed = 0usize;
    let mut non_empty = 0usize;
    for rec in rows.iter().take(5000) {
        let raw = rec.get(idx).unwrap_or("").trim();
        if raw.is_empty() {
            continue;
        }
        non_empty += 1;
        if parse_numeric(raw).is_some() {
            parsed += 1;
        }
    }
    non_empty >= 20 && (parsed as f64 / non_empty as f64) >= 0.8
}

fn approx_distinct_count(rows: &[StringRecord], idx: usize, max_rows: usize) -> (usize, usize) {
    let mut vals = HashSet::<String>::new();
    let mut non_empty = 0usize;
    for rec in rows.iter().take(max_rows) {
        let raw = rec.get(idx).unwrap_or("").trim();
        if raw.is_empty() {
            continue;
        }
        non_empty += 1;
        vals.insert(raw.to_ascii_lowercase());
    }
    (vals.len(), non_empty)
}

fn contains_any(s: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| s.contains(n))
}

fn compose_group_key(
    rec: &StringRecord,
    group_idxs: &[usize],
    group_names: &[String],
    normalize_text_groups: bool,
) -> String {
    group_idxs
        .iter()
        .enumerate()
        .map(|(i, idx)| {
            let raw = rec.get(*idx).unwrap_or("").trim();
            let mut v = if normalize_text_groups
                && group_names
                    .get(i)
                    .map(|n| should_normalize_group_column(n))
                    .unwrap_or(false)
            {
                normalize_group_value(raw)
            } else {
                raw.to_string()
            };
            if is_effectively_blank(&v) {
                v = "(blank)".to_string();
            }
            v
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn infer_numeric_driver_agg(col_name: &str) -> DriverAgg {
    let n = col_name.to_ascii_lowercase();
    if n.starts_with("avg_")
        || n.contains("rate")
        || n.contains("ratio")
        || n.contains("pct")
        || n.contains("percent")
        || n.contains("conversion")
        || n.contains("price")
        || n.contains("unit_price")
        || n.contains("arpu")
        || n.contains("cpc")
        || n.contains("cpm")
    {
        DriverAgg::Mean
    } else {
        DriverAgg::Sum
    }
}

fn is_count_like_numeric_column(col_name: &str) -> bool {
    let n = col_name.to_ascii_lowercase();
    n.starts_with("count_") || n.ends_with("_count") || n.contains("_count_")
}

fn is_indicator_like_numeric_column(col_name: &str) -> bool {
    let n = col_name.to_ascii_lowercase();
    n.starts_with("is_")
        || n.starts_with("has_")
        || n.contains("flag")
        || n.contains("plan")
        || n.contains("indicator")
        || n.contains("bool")
        || n.contains("boolean")
}

fn auto_select_numeric_drivers(
    headers: &StringRecord,
    rows: &[StringRecord],
    metric_idx: usize,
    date_idx: usize,
    top_n: usize,
    dedup_corr_threshold: Option<f64>,
) -> Vec<(String, usize)> {
    let mut scored = Vec::<(String, usize, f64)>::new();
    for (idx, name) in headers.iter().enumerate() {
        if idx == metric_idx || idx == date_idx {
            continue;
        }
        let lname = name.to_lowercase();
        if lname == "id"
            || lname.ends_with("_id")
            || lname.contains("date")
            || lname.contains("time")
            || is_count_like_numeric_column(&lname)
        {
            continue;
        }
        let (distinct_count, non_empty) = approx_distinct_count(rows, idx, 5000);
        if non_empty < 20 {
            continue;
        }
        if distinct_count <= 3 || (is_indicator_like_numeric_column(&lname) && distinct_count <= 10) {
            continue;
        }
        let mut xs = Vec::<f64>::new();
        let mut ys = Vec::<f64>::new();
        for rec in rows.iter().take(5000) {
            let x = parse_numeric(rec.get(idx).unwrap_or("").trim());
            let y = parse_numeric(rec.get(metric_idx).unwrap_or("").trim());
            if let (Some(x), Some(y)) = (x, y) {
                xs.push(x);
                ys.push(y);
            }
        }
        if xs.len() < 20 {
            continue;
        }
        let corr = pearson_corr(&xs, &ys).abs();
        if corr.is_finite() {
            scored.push((name.to_string(), idx, corr));
        }
    }
    scored.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut selected = Vec::<(String, usize)>::new();
    for (name, idx, _corr_to_metric) in scored {
        let is_duplicate = dedup_corr_threshold
            .map(|thr| {
                selected.iter().any(|(_, prev_idx)| {
                    let c = corr_between_columns(rows, idx, *prev_idx).abs();
                    c.is_finite() && c >= thr
                })
            })
            .unwrap_or(false);
        if is_duplicate {
            continue;
        }
        selected.push((name, idx));
        if selected.len() >= top_n {
            break;
        }
    }
    selected
        .into_iter()
        .map(|(n, i)| (n, i))
        .collect::<Vec<_>>()
}

fn corr_between_columns(rows: &[StringRecord], a_idx: usize, b_idx: usize) -> f64 {
    let mut xs = Vec::<f64>::new();
    let mut ys = Vec::<f64>::new();
    for rec in rows.iter().take(5000) {
        let x = parse_numeric(rec.get(a_idx).unwrap_or("").trim());
        let y = parse_numeric(rec.get(b_idx).unwrap_or("").trim());
        if let (Some(x), Some(y)) = (x, y) {
            xs.push(x);
            ys.push(y);
        }
    }
    if xs.len() < 20 {
        0.0
    } else {
        pearson_corr(&xs, &ys)
    }
}

fn pearson_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let mx = xs.iter().take(n).sum::<f64>() / n_f;
    let my = ys.iter().take(n).sum::<f64>() / n_f;
    let mut num = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        num += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let den = (vx * vy).sqrt();
    if den <= 1e-12 { 0.0 } else { num / den }
}

fn parse_period_cfg_from_investigate(
    args: &InvestigateArgs,
    date_column: String,
) -> Result<PeriodCompareConfig> {
    let explicit_windows = args.current_start.is_some()
        || args.current_end.is_some()
        || args.previous_start.is_some()
        || args.previous_end.is_some();
    let derived_windows = args.time_grain.is_some() || args.period.is_some() || args.anchor_date.is_some();

    if explicit_windows && derived_windows {
        return Err(anyhow!(
            "use either explicit windows (--current-start/--current-end/--previous-start/--previous-end) OR derived windows (--time-grain/--period/--anchor-date), not both"
        ));
    }

    let (time_grain, period, current_start, current_end, previous_start, previous_end) = if explicit_windows {
        let cs = parse_date_arg(args.current_start.as_deref(), "--current-start is required")?;
        let ce = parse_date_arg(args.current_end.as_deref(), "--current-end is required")?;
        let ps = parse_date_arg(args.previous_start.as_deref(), "--previous-start is required")?;
        let pe = parse_date_arg(args.previous_end.as_deref(), "--previous-end is required")?;
        (None, None, cs, ce, ps, pe)
    } else {
        let grain = args.time_grain.unwrap_or(TimeGrain::Month);
        let p = args.period.unwrap_or(PeriodPreset::Last);
        let anchor = match args.anchor_date.as_deref() {
            Some(raw) => chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d")
                .map_err(|_| anyhow!("invalid date '{}'; expected YYYY-MM-DD", raw))?,
            None => chrono::Utc::now().date_naive(),
        };
        let ((cs, ce), (ps, pe)) = derive_period_windows(anchor, grain, p);
        (Some(grain), Some(p), cs, ce, ps, pe)
    };

    Ok(PeriodCompareConfig {
        date_column,
        time_grain,
        period,
        current_start,
        current_end,
        previous_start,
        previous_end,
    })
}

fn parse_period_compare_config(args: &AnalyzeArgs) -> Result<Option<PeriodCompareConfig>> {
    let any_period_flag = args.date_column.is_some()
        || args.time_grain.is_some()
        || args.period.is_some()
        || args.anchor_date.is_some()
        || args.current_start.is_some()
        || args.current_end.is_some()
        || args.previous_start.is_some()
        || args.previous_end.is_some();
    if !any_period_flag {
        return Ok(None);
    }

    let date_column = args
        .date_column
        .clone()
        .ok_or_else(|| anyhow!("--date-column is required when using period comparison flags"))?;
    let explicit_windows = args.current_start.is_some()
        || args.current_end.is_some()
        || args.previous_start.is_some()
        || args.previous_end.is_some();
    let derived_windows =
        args.time_grain.is_some() || args.period.is_some() || args.anchor_date.is_some();

    if explicit_windows && derived_windows {
        return Err(anyhow!(
            "use either explicit windows (--current-start/--current-end/--previous-start/--previous-end) OR derived windows (--time-grain/--period/--anchor-date), not both"
        ));
    }

    let (time_grain, period, current_start, current_end, previous_start, previous_end) =
        if explicit_windows {
            let cs = parse_date_arg(
                args.current_start.as_deref(),
                "--current-start is required for period comparison",
            )?;
            let ce = parse_date_arg(
                args.current_end.as_deref(),
                "--current-end is required for period comparison",
            )?;
            let ps = parse_date_arg(
                args.previous_start.as_deref(),
                "--previous-start is required for period comparison",
            )?;
            let pe = parse_date_arg(
                args.previous_end.as_deref(),
                "--previous-end is required for period comparison",
            )?;
            (None, None, cs, ce, ps, pe)
        } else {
            let grain = args.time_grain.ok_or_else(|| {
                anyhow!("--time-grain is required when using derived period comparison")
            })?;
            let period = args.period.unwrap_or(PeriodPreset::Current);
            let anchor = match args.anchor_date.as_deref() {
                Some(raw) => chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d")
                    .map_err(|_| anyhow!("invalid date '{}'; expected YYYY-MM-DD", raw))?,
                None => chrono::Utc::now().date_naive(),
            };
            let ((cs, ce), (ps, pe)) = derive_period_windows(anchor, grain, period);
            (Some(grain), Some(period), cs, ce, ps, pe)
        };

    if current_start > current_end {
        return Err(anyhow!("current period start must be <= end"));
    }
    if previous_start > previous_end {
        return Err(anyhow!("previous period start must be <= end"));
    }

    Ok(Some(PeriodCompareConfig {
        date_column,
        time_grain,
        period,
        current_start,
        current_end,
        previous_start,
        previous_end,
    }))
}

fn parse_date_arg(v: Option<&str>, missing_msg: &str) -> Result<chrono::NaiveDate> {
    let raw = v.ok_or_else(|| anyhow!("{}", missing_msg))?;
    chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d")
        .map_err(|_| anyhow!("invalid date '{}'; expected YYYY-MM-DD", raw))
}

fn derive_period_windows(
    anchor: chrono::NaiveDate,
    grain: TimeGrain,
    period: PeriodPreset,
) -> (
    (chrono::NaiveDate, chrono::NaiveDate),
    (chrono::NaiveDate, chrono::NaiveDate),
) {
    let base = period_containing(anchor, grain);
    let current = match period {
        PeriodPreset::Current => base,
        PeriodPreset::Previous | PeriodPreset::Last => period_before(base.0, grain),
    };
    let previous = period_before(current.0, grain);
    (current, previous)
}

fn period_containing(
    d: chrono::NaiveDate,
    grain: TimeGrain,
) -> (chrono::NaiveDate, chrono::NaiveDate) {
    match grain {
        TimeGrain::Day => (d, d),
        TimeGrain::Week => {
            let start = d - chrono::Duration::days(d.weekday().num_days_from_monday() as i64);
            (start, start + chrono::Duration::days(6))
        }
        TimeGrain::Month => {
            let start = chrono::NaiveDate::from_ymd_opt(d.year(), d.month(), 1).unwrap_or(d);
            let (ny, nm) = if d.month() == 12 {
                (d.year() + 1, 1)
            } else {
                (d.year(), d.month() + 1)
            };
            let next_start = chrono::NaiveDate::from_ymd_opt(ny, nm, 1).unwrap_or(start);
            (start, next_start - chrono::Duration::days(1))
        }
        TimeGrain::Year => {
            let start = chrono::NaiveDate::from_ymd_opt(d.year(), 1, 1).unwrap_or(d);
            let next_start = chrono::NaiveDate::from_ymd_opt(d.year() + 1, 1, 1).unwrap_or(start);
            (start, next_start - chrono::Duration::days(1))
        }
    }
}

fn period_before(
    current_start: chrono::NaiveDate,
    grain: TimeGrain,
) -> (chrono::NaiveDate, chrono::NaiveDate) {
    match grain {
        TimeGrain::Day => {
            let d = current_start - chrono::Duration::days(1);
            (d, d)
        }
        TimeGrain::Week => {
            let end = current_start - chrono::Duration::days(1);
            let start = end - chrono::Duration::days(6);
            (start, end)
        }
        TimeGrain::Month => {
            let end = current_start - chrono::Duration::days(1);
            let start = chrono::NaiveDate::from_ymd_opt(end.year(), end.month(), 1).unwrap_or(end);
            (start, end)
        }
        TimeGrain::Year => {
            let end = current_start - chrono::Duration::days(1);
            let start = chrono::NaiveDate::from_ymd_opt(end.year(), 1, 1).unwrap_or(end);
            (start, end)
        }
    }
}

fn default_analyze_out(args: &AnalyzeArgs) -> PathBuf {
    let base_stem = args
        .input
        .as_ref()
        .and_then(|p| p.file_stem())
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "analysis".to_string());
    let stem_norm = base_stem.to_ascii_lowercase();
    let final_stem = if stem_norm.starts_with("analyze_") {
        base_stem
    } else {
        format!("analyze_{}", base_stem)
    };
    let base = PathBuf::from("artifacts").join(final_stem);

    match args.output_format {
        OutputFormat::Md | OutputFormat::Both => base.with_extension("md"),
        OutputFormat::Json => base.with_extension("json"),
        OutputFormat::Html => base.with_extension("html"),
    }
}

fn default_analyze_investigate_out(args: &InvestigateArgs) -> PathBuf {
    let base_stem = args
        .input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "analysis".to_string());
    let stem_norm = base_stem.to_ascii_lowercase();
    let final_stem = if stem_norm.starts_with("investigate_") {
        base_stem
    } else {
        format!("investigate_{}", base_stem)
    };
    let base = PathBuf::from("artifacts").join(final_stem);
    match args.output_format {
        InvestigateOutputFormat::Md | InvestigateOutputFormat::Both => base.with_extension("md"),
        InvestigateOutputFormat::Json => base.with_extension("json"),
    }
}

fn default_analyze_drivers_out(args: &AnalyzeDriversArgs) -> PathBuf {
    let base_stem = args
        .input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "analysis".to_string());
    let stem_norm = base_stem.to_ascii_lowercase();
    let final_stem = if stem_norm.starts_with("drivers_") {
        base_stem
    } else {
        format!("drivers_{}", base_stem)
    };
    let base = PathBuf::from("artifacts").join(final_stem);
    match args.output_format {
        InvestigateOutputFormat::Md | InvestigateOutputFormat::Both => base.with_extension("md"),
        InvestigateOutputFormat::Json => base.with_extension("json"),
    }
}

fn ensure_parent_dir(path: &PathBuf) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
struct SuggestColumn {
    name: String,
    non_empty: u64,
    fill_pct: f64,
    distinct_count: usize,
    numeric_ratio: f64,
    date_ratio: f64,
    inferred_role: String,
    top_values: Vec<(String, u64)>,
}

#[derive(Debug, Serialize)]
struct AnalyzeSuggestReport {
    input: String,
    sampled_rows: usize,
    sample_mode: String,
    sample_seed: u64,
    profile_name: String,
    suggested_group_by: Vec<String>,
    suggested_metrics: Vec<String>,
    suggested_rank_by: Option<String>,
    suggested_date_column: Option<String>,
    warnings: Vec<String>,
    columns: Vec<SuggestColumn>,
}

fn run_analyze_suggest(args: AnalyzeSuggestArgs) -> Result<()> {
    let report = analyze_suggest_csv(&args)?;
    let profile_toml = build_suggested_profile_toml(
        &report,
        &args.profile_name,
        args.auto_group_k,
        args.max_metrics,
    );
    let profile_path = args
        .out_profile
        .clone()
        .unwrap_or_else(|| args.out.with_extension("toml"));
    fs::write(&profile_path, profile_toml)?;

    match args.output_format {
        SuggestOutputFormat::Md => {
            fs::write(&args.out, suggest_report_markdown(&report, &profile_path))?;
            println!("Analyze suggest report (markdown) written to {}", args.out.display());
        }
        SuggestOutputFormat::Json => {
            fs::write(&args.out, serde_json::to_string_pretty(&report)?)?;
            println!("Analyze suggest report (json) written to {}", args.out.display());
        }
        SuggestOutputFormat::Both => {
            fs::write(&args.out, suggest_report_markdown(&report, &profile_path))?;
            let json_path = args.out.with_extension("json");
            fs::write(&json_path, serde_json::to_string_pretty(&report)?)?;
            println!("Analyze suggest report (markdown) written to {}", args.out.display());
            println!("Analyze suggest report (json) written to {}", json_path.display());
        }
    }
    println!("Suggested profile TOML written to {}", profile_path.display());
    Ok(())
}

fn analyze_suggest_csv(args: &AnalyzeSuggestArgs) -> Result<AnalyzeSuggestReport> {
    let mut rdr = csv::Reader::from_path(&args.input)?;
    let headers = rdr.headers()?.clone();
    let col_count = headers.len();
    if col_count == 0 {
        return Err(anyhow!("input CSV has no columns"));
    }

    let mut sampled_rows = 0usize;
    let mut non_empty = vec![0u64; col_count];
    let mut numeric_ok = vec![0u64; col_count];
    let mut date_ok = vec![0u64; col_count];
    let mut distinct = vec![HashSet::<String>::new(); col_count];
    let mut counts = vec![HashMap::<String, u64>::new(); col_count];

    let mut sampled_records: Vec<StringRecord> = Vec::new();
    match args.sample_mode {
        SampleMode::Head => {
            for rec in rdr.records().take(args.sample_rows) {
                sampled_records.push(rec?);
            }
        }
        SampleMode::Random => {
            let mut state = args.sample_seed;
            for (idx, rec) in rdr.records().enumerate() {
                let rec = rec?;
                if sampled_records.len() < args.sample_rows {
                    sampled_records.push(rec);
                    continue;
                }
                let j = pseudo_rand_below(&mut state, (idx + 1) as u64) as usize;
                if j < args.sample_rows {
                    sampled_records[j] = rec;
                }
            }
        }
    }

    for rec in &sampled_records {
        sampled_rows += 1;
        for i in 0..col_count {
            let raw = rec.get(i).unwrap_or("").trim();
            if raw.is_empty() {
                continue;
            }
            non_empty[i] += 1;
            if parse_numeric(raw).is_some() {
                numeric_ok[i] += 1;
            }
            if parse_date_like(raw).is_some() {
                date_ok[i] += 1;
            }

            if distinct[i].len() < 2000 {
                distinct[i].insert(raw.to_string());
            }
            if counts[i].len() < 300 || counts[i].contains_key(raw) {
                *counts[i].entry(raw.to_string()).or_insert(0) += 1;
            }
        }
    }

    if sampled_rows == 0 {
        return Err(anyhow!("input CSV has no rows"));
    }

    let mut columns = Vec::with_capacity(col_count);
    for i in 0..col_count {
        let name = headers.get(i).unwrap_or("").to_string();
        let ne = non_empty[i];
        let fill_pct = (ne as f64 / sampled_rows as f64) * 100.0;
        let numeric_ratio = if ne == 0 {
            0.0
        } else {
            numeric_ok[i] as f64 / ne as f64
        };
        let date_ratio = if ne == 0 {
            0.0
        } else {
            date_ok[i] as f64 / ne as f64
        };
        let distinct_count = distinct[i].len();
        let inferred_role = infer_column_role(&name, fill_pct, distinct_count, numeric_ratio, date_ratio);
        let mut top_values = counts[i]
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect::<Vec<_>>();
        top_values.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        top_values.truncate(5);

        columns.push(SuggestColumn {
            name,
            non_empty: ne,
            fill_pct,
            distinct_count,
            numeric_ratio,
            date_ratio,
            inferred_role,
            top_values,
        });
    }

    let mut suggested_group_by = columns
        .iter()
        .filter(|c| c.inferred_role == "dimension")
        .cloned()
        .collect::<Vec<_>>();
    suggested_group_by.sort_by(|a, b| {
        let a_penalty = (a.distinct_count as i64 - 12).abs();
        let b_penalty = (b.distinct_count as i64 - 12).abs();
        a_penalty
            .cmp(&b_penalty)
            .then_with(|| b.fill_pct.partial_cmp(&a.fill_pct).unwrap_or(std::cmp::Ordering::Equal))
    });
    let suggested_group_by = suggested_group_by
        .into_iter()
        .take(args.auto_group_k)
        .map(|c| c.name)
        .collect::<Vec<_>>();

    let mut suggested_metrics = columns
        .iter()
        .filter(|c| c.inferred_role == "metric")
        .cloned()
        .collect::<Vec<_>>();
    suggested_metrics.sort_by(|a, b| {
        metric_priority(&b.name)
            .cmp(&metric_priority(&a.name))
            .then_with(|| b.non_empty.cmp(&a.non_empty))
    });
    let suggested_metrics = suggested_metrics
        .into_iter()
        .take(args.max_metrics)
        .map(|c| c.name)
        .collect::<Vec<_>>();

    let suggested_rank_by = suggested_metrics.first().cloned();
    let suggested_date_column = columns
        .iter()
        .filter(|c| c.inferred_role == "date")
        .max_by(|a, b| a.fill_pct.partial_cmp(&b.fill_pct).unwrap_or(std::cmp::Ordering::Equal))
        .map(|c| c.name.clone());

    let mut warnings = Vec::new();
    if suggested_group_by.is_empty() {
        warnings.push("No strong dimension columns detected. Pass --group-by manually.".to_string());
    }
    if suggested_metrics.is_empty() {
        warnings.push("No strong metric columns detected. Pass --metrics manually.".to_string());
    }
    for c in columns.iter().filter(|c| c.fill_pct < 30.0 && c.inferred_role == "dimension") {
        warnings.push(format!(
            "Dimension '{}' has low fill rate ({:.1}%).",
            c.name, c.fill_pct
        ));
    }

    Ok(AnalyzeSuggestReport {
        input: args.input.display().to_string(),
        sampled_rows,
        sample_mode: match args.sample_mode {
            SampleMode::Head => "head".to_string(),
            SampleMode::Random => "random".to_string(),
        },
        sample_seed: args.sample_seed,
        profile_name: args.profile_name.clone(),
        suggested_group_by,
        suggested_metrics,
        suggested_rank_by,
        suggested_date_column,
        warnings,
        columns,
    })
}

fn infer_column_role(
    name: &str,
    fill_pct: f64,
    distinct_count: usize,
    numeric_ratio: f64,
    date_ratio: f64,
) -> String {
    let n = name.to_lowercase();
    if date_ratio >= 0.9 || n == "date" || n.ends_with("_date") || n.contains("timestamp") {
        return "date".to_string();
    }
    if numeric_ratio >= 0.85
        && fill_pct >= 20.0
        && distinct_count <= 20
        && (n.contains("tier")
            || n.contains("plan")
            || n.contains("flag")
            || n.contains("status")
            || n.contains("bucket")
            || n.contains("class"))
    {
        return "dimension".to_string();
    }
    let id_like = n == "id"
        || n.ends_with("_id")
        || n.ends_with("_uuid")
        || n.contains("uuid")
        || n.ends_with("_url");
    if numeric_ratio >= 0.85 && !id_like {
        return "metric".to_string();
    }
    if !id_like && fill_pct >= 20.0 && (2..=80).contains(&distinct_count) {
        return "dimension".to_string();
    }
    if numeric_ratio >= 0.85 {
        return "numeric_other".to_string();
    }
    "text_other".to_string()
}

fn metric_priority(name: &str) -> i32 {
    let n = name.to_lowercase();
    if n.contains("revenue") || n.contains("gmv") || n.contains("sales") || n.contains("amount") {
        return 3;
    }
    if n.contains("cost") || n.contains("profit") || n.contains("margin") {
        return 2;
    }
    if n.contains("qty") || n.contains("quantity") || n.contains("count") || n.contains("orders")
    {
        return 1;
    }
    0
}

fn build_suggested_profile_toml(
    report: &AnalyzeSuggestReport,
    profile_name: &str,
    auto_group_k: usize,
    max_metrics: usize,
) -> String {
    let mut s = String::new();
    s.push_str(&format!("[profiles.{}]\n", profile_name));
    if !report.suggested_group_by.is_empty() {
        s.push_str(&format!(
            "group_by = [{}]\n",
            report
                .suggested_group_by
                .iter()
                .map(|c| format!("\"{}\"", c))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !report.suggested_metrics.is_empty() {
        s.push_str(&format!(
            "metrics = [{}]\n",
            report
                .suggested_metrics
                .iter()
                .take(max_metrics)
                .map(|c| format!("\"{}\"", c))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(rank_by) = &report.suggested_rank_by {
        s.push_str(&format!("rank_by = \"{}\"\n", rank_by));
    }
    s.push_str("top = 15\n");
    s.push_str("min_records = 10\n");
    s.push_str(&format!("auto_group_k = {}\n", auto_group_k));
    s
}

fn suggest_report_markdown(report: &AnalyzeSuggestReport, profile_path: &PathBuf) -> String {
    let mut md = String::new();
    md.push_str("# Analyze Suggest Report\n\n");
    md.push_str(&format!("- Input: {}\n", report.input));
    md.push_str(&format!("- Sampled rows: {}\n", report.sampled_rows));
    md.push_str(&format!(
        "- Sample mode: {} (seed={})\n",
        report.sample_mode, report.sample_seed
    ));
    md.push_str(&format!("- Suggested profile name: `{}`\n", report.profile_name));
    md.push_str(&format!("- Suggested profile path: {}\n\n", profile_path.display()));

    md.push_str("## Suggested Columns\n\n");
    md.push_str(&format!(
        "- group_by: {}\n",
        if report.suggested_group_by.is_empty() {
            "(none)".to_string()
        } else {
            report.suggested_group_by.join(", ")
        }
    ));
    md.push_str(&format!(
        "- metrics: {}\n",
        if report.suggested_metrics.is_empty() {
            "(none)".to_string()
        } else {
            report.suggested_metrics.join(", ")
        }
    ));
    md.push_str(&format!(
        "- rank_by: {}\n",
        report
            .suggested_rank_by
            .clone()
            .unwrap_or_else(|| "(none)".to_string())
    ));
    md.push_str(&format!(
        "- date_column: {}\n\n",
        report
            .suggested_date_column
            .clone()
            .unwrap_or_else(|| "(none)".to_string())
    ));

    if !report.warnings.is_empty() {
        md.push_str("## Warnings\n\n");
        for w in &report.warnings {
            md.push_str(&format!("- {}\n", w));
        }
        md.push('\n');
    }

    md.push_str("## Column Profile\n\n");
    md.push_str("| Column | Role | Fill % | Distinct | Numeric Ratio | Date Ratio | Top Values |\n");
    md.push_str("|---|---|---:|---:|---:|---:|---|\n");
    for c in &report.columns {
        let top_vals = c
            .top_values
            .iter()
            .map(|(v, n)| format!("{} ({})", v.replace('|', "\\|"), n))
            .collect::<Vec<_>>()
            .join("; ");
        md.push_str(&format!(
            "| {} | {} | {:.1}% | {} | {:.2} | {:.2} | {} |\n",
            c.name,
            c.inferred_role,
            c.fill_pct,
            c.distinct_count,
            c.numeric_ratio,
            c.date_ratio,
            top_vals
        ));
    }
    md
}

fn run_analyze_compare(args: AnalyzeCompareArgs) -> Result<()> {
    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| default_analyze_compare_out(args.output_format));
    ensure_parent_dir(&out_path)?;

    let base_txt = fs::read_to_string(&args.base)
        .map_err(|e| anyhow!("failed to read base json '{}': {}", args.base.display(), e))?;
    let new_txt = fs::read_to_string(&args.new)
        .map_err(|e| anyhow!("failed to read new json '{}': {}", args.new.display(), e))?;
    let base: serde_json::Value = serde_json::from_str(&base_txt).map_err(|e| {
        anyhow!(
            "failed to parse base json '{}': {}",
            args.base.display(),
            e
        )
    })?;
    let new: serde_json::Value = serde_json::from_str(&new_txt)
        .map_err(|e| anyhow!("failed to parse new json '{}': {}", args.new.display(), e))?;

    let base_records = base.get("records").and_then(|x| x.as_u64()).unwrap_or(0);
    let new_records = new.get("records").and_then(|x| x.as_u64()).unwrap_or(0);
    let base_segments = base.get("segments").and_then(|x| x.as_u64()).unwrap_or(0);
    let new_segments = new.get("segments").and_then(|x| x.as_u64()).unwrap_or(0);
    let base_top5_count = base.get("top5_count").and_then(|x| x.as_u64()).unwrap_or(0);
    let new_top5_count = new.get("top5_count").and_then(|x| x.as_u64()).unwrap_or(0);
    let base_top5_pct = pct(base_top5_count, base_records);
    let new_top5_pct = pct(new_top5_count, new_records);
    let primary_metric = new
        .get("primary_metric")
        .and_then(|x| x.as_str())
        .or_else(|| base.get("primary_metric").and_then(|x| x.as_str()))
        .unwrap_or("primary_metric")
        .to_string();

    let base_map = groups_to_map(&base, &primary_metric);
    let new_map = groups_to_map(&new, &primary_metric);
    let mut keys = base_map.keys().cloned().collect::<HashSet<_>>();
    keys.extend(new_map.keys().cloned());

    let mut movers = keys
        .into_iter()
        .map(|k| {
            let (bc, bs, bp) = base_map.get(&k).copied().unwrap_or((0, 0.0, 0.0));
            let (nc, ns, np) = new_map.get(&k).copied().unwrap_or((0, 0.0, 0.0));
            let d_share = ns - bs;
            let d_primary = np - bp;
            (k, bc, nc, bs, ns, d_share, bp, np, d_primary)
        })
        .collect::<Vec<_>>();
    movers.sort_by(|a, b| {
        b.5.abs()
            .partial_cmp(&a.5.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut md = String::new();
    md.push_str("# Analysis Comparison\n\n");
    md.push_str(&format!("- Base: {}\n", args.base.display()));
    md.push_str(&format!("- New: {}\n", args.new.display()));
    md.push_str(&format!("- Base records: {}\n", base_records));
    md.push_str(&format!("- New records: {}\n", new_records));
    md.push_str(&format!("- Base segments: {}\n", base_segments));
    md.push_str(&format!("- New segments: {}\n", new_segments));
    md.push('\n');

    md.push_str("## Executive Delta\n\n");
    md.push_str(&format!(
        "- Top-5 concentration changed from {:.1}% to {:.1}% ({:+.1} pp).\n",
        base_top5_pct,
        new_top5_pct,
        new_top5_pct - base_top5_pct
    ));
    md.push_str(&format!(
        "- Segment count changed from {} to {} ({:+}).\n",
        base_segments,
        new_segments,
        new_segments as i64 - base_segments as i64
    ));
    md.push('\n');

    md.push_str("## Biggest Movers (by record share)\n\n");
    md.push_str(&format!(
        "| Segment | Base Records | New Records | Base Share | New Share | Delta Share (pp) | Base {} | New {} | Delta {} |\n",
        primary_metric, primary_metric, primary_metric
    ));
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (seg, bc, nc, bs, ns, ds, bp, np, dp) in movers.iter().take(args.top_movers) {
        md.push_str(&format!(
            "| {} | {} | {} | {:.1}% | {:.1}% | {:+.1} | {} | {} | {} |\n",
            seg.replace('|', "\\|"),
            bc,
            nc,
            bs,
            ns,
            ds,
            fmt_num(*bp, 2),
            fmt_num(*np, 2),
            fmt_num(*dp, 2)
        ));
    }

    let movers_json = movers
        .iter()
        .take(args.top_movers)
        .map(|(seg, bc, nc, bs, ns, ds, bp, np, dp)| {
            serde_json::json!({
                "segment": seg,
                "base_records": bc,
                "new_records": nc,
                "base_share_pct": bs,
                "new_share_pct": ns,
                "delta_share_pp": ds,
                "base_primary_metric_value": bp,
                "new_primary_metric_value": np,
                "delta_primary_metric_value": dp
            })
        })
        .collect::<Vec<_>>();

    let json_out = serde_json::json!({
        "base": args.base.display().to_string(),
        "new": args.new.display().to_string(),
        "base_records": base_records,
        "new_records": new_records,
        "base_segments": base_segments,
        "new_segments": new_segments,
        "primary_metric": primary_metric,
        "top5_concentration": {
            "base_pct": base_top5_pct,
            "new_pct": new_top5_pct,
            "delta_pp": new_top5_pct - base_top5_pct
        },
        "segment_count_delta": {
            "base": base_segments,
            "new": new_segments,
            "delta": new_segments as i64 - base_segments as i64
        },
        "top_movers_limit": args.top_movers,
        "movers": movers_json
    });

    match args.output_format {
        CompareOutputFormat::Md => {
            fs::write(&out_path, md)?;
            println!("Comparison report (markdown) written to {}", out_path.display());
        }
        CompareOutputFormat::Html => {
            fs::write(&out_path, markdown_to_html(&md))?;
            println!("Comparison report (html) written to {}", out_path.display());
        }
        CompareOutputFormat::Json => {
            fs::write(&out_path, serde_json::to_string_pretty(&json_out)?)?;
            println!("Comparison report (json) written to {}", out_path.display());
        }
        CompareOutputFormat::Both => {
            fs::write(&out_path, md)?;
            let json_path = out_path.with_extension("json");
            fs::write(&json_path, serde_json::to_string_pretty(&json_out)?)?;
            println!("Comparison report (markdown) written to {}", out_path.display());
            println!("Comparison report (json) written to {}", json_path.display());
        }
    }
    Ok(())
}

fn default_analyze_compare_out(format: CompareOutputFormat) -> PathBuf {
    match format {
        CompareOutputFormat::Md | CompareOutputFormat::Both => {
            PathBuf::from("artifacts/analysis_compare.md")
        }
        CompareOutputFormat::Html => PathBuf::from("artifacts/analysis_compare.html"),
        CompareOutputFormat::Json => PathBuf::from("artifacts/analysis_compare.json"),
    }
}

fn groups_to_map(v: &serde_json::Value, primary_metric: &str) -> HashMap<String, (u64, f64, f64)> {
    let mut out = HashMap::new();
    let groups = v
        .get("groups")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    for g in groups {
        let name = g
            .get("group")
            .and_then(|x| x.as_str())
            .unwrap_or("(unknown)")
            .to_string();
        let count = g.get("count").and_then(|x| x.as_u64()).unwrap_or(0);
        let share = g.get("count_share_pct").and_then(|x| x.as_f64()).unwrap_or(0.0);
        let primary = g
            .get(primary_metric)
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        out.insert(name, (count, share, primary));
    }
    out
}

fn pct(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        (num as f64 / den as f64) * 100.0
    }
}

fn materialize_analyze_input(args: &AnalyzeArgs) -> Result<(PathBuf, Option<PathBuf>)> {
    if let Some(path) = &args.input {
        if args.postgres_url.is_some()
            || args.query.is_some()
            || args.query_file.is_some()
            || args.postgres_ca_file.is_some()
            || args.postgres_ssl_mode != PostgresSslMode::Prefer
        {
            return Err(anyhow!(
                "choose exactly one input source: --input <csv> OR --postgres-url + (--query | --query-file)"
            ));
        }
        return Ok((path.clone(), None));
    }

    let effective_pg_url = args
        .postgres_url
        .clone()
        .or_else(|| std::env::var("DATABASE_URL").ok());
    match (&args.input, effective_pg_url.as_deref(), &args.query, &args.query_file) {
        (None, Some(url), Some(q), None) => {
            let path = postgres_query_to_temp_csv(
                url,
                q,
                args.postgres_ssl_mode,
                args.postgres_ca_file.as_ref(),
            )?;
            Ok((path.clone(), Some(path)))
        }
        (None, Some(url), None, Some(query_file)) => {
            let q = fs::read_to_string(query_file).map_err(|e| {
                anyhow!(
                    "failed to read query file '{}': {}",
                    query_file.display(),
                    e
                )
            })?;
            let path = postgres_query_to_temp_csv(
                url,
                &q,
                args.postgres_ssl_mode,
                args.postgres_ca_file.as_ref(),
            )?;
            Ok((path.clone(), Some(path)))
        }
        (None, Some(_), None, None) => Err(anyhow!(
            "postgres analyze requires one of --query or --query-file"
        )),
        (None, None, Some(_), _) | (None, None, _, Some(_)) => {
            Err(anyhow!(
                "--query/--query-file require --postgres-url (or DATABASE_URL env var)"
            ))
        }
        _ => Err(anyhow!(
            "choose exactly one input source: --input <csv> OR --postgres-url + (--query | --query-file)"
        )),
    }
}

fn postgres_query_to_temp_csv(
    postgres_url: &str,
    query: &str,
    ssl_mode: PostgresSslMode,
    ca_file: Option<&PathBuf>,
) -> Result<PathBuf> {
    let mut client = match connect_postgres(postgres_url, ssl_mode, ca_file) {
        Ok(c) => c,
        Err(e) => return Err(anyhow!("failed to connect to postgres: {}", e)),
    };
    let copy_sql = format!(
        "COPY ({}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)",
        query.trim()
    );
    let mut reader = client
        .copy_out(copy_sql.as_str())
        .map_err(|e| anyhow!("failed to run COPY export for query: {}", e))?;

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let tmp_path = std::env::temp_dir().join(format!("factorlens_analyze_{}.csv", ts));
    let mut file = fs::File::create(&tmp_path).map_err(|e| {
        anyhow!(
            "failed to create temporary csv '{}': {}",
            tmp_path.display(),
            e
        )
    })?;
    std::io::copy(&mut reader, &mut file)
        .map_err(|e| anyhow!("failed writing postgres result csv: {}", e))?;
    file.flush()?;
    Ok(tmp_path)
}

fn connect_postgres(
    postgres_url: &str,
    ssl_mode: PostgresSslMode,
    ca_file: Option<&PathBuf>,
) -> Result<Client> {
    if ssl_mode == PostgresSslMode::Disable {
        return Client::connect(postgres_url, NoTls)
            .map_err(|e| anyhow!("non-tls connect error: {}", e));
    }

    let mut root_store = rustls::RootCertStore::empty();
    let certs = rustls_native_certs::load_native_certs();
    for cert in certs.certs {
        if root_store.add(cert).is_err() {
            // Skip invalid certs and continue with remaining roots.
        }
    }
    if let Some(path) = ca_file {
        let file = fs::File::open(path).map_err(|e| {
            anyhow!(
                "failed to open --postgres-ca-file '{}': {}",
                path.display(),
                e
            )
        })?;
        let mut reader = BufReader::new(file);
        for cert in rustls_pemfile::certs(&mut reader) {
            let cert = cert.map_err(|e| {
                anyhow!(
                    "failed to parse PEM cert in '{}': {}",
                    path.display(),
                    e
                )
            })?;
            if root_store.add(cert).is_err() {
                // Ignore invalid certs; continue loading others.
            }
        }
    }
    let tls_config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    let tls_connector = tokio_rustls::TlsConnector::from(Arc::new(tls_config));
    let tls_connector = MakeTlsConnector::new(tls_connector);

    match Client::connect(postgres_url, tls_connector) {
        Ok(c) => Ok(c),
        Err(tls_err) => {
            if ssl_mode == PostgresSslMode::Require {
                Err(anyhow!("tls connect error: {}", tls_err))
            } else {
                Client::connect(postgres_url, NoTls).map_err(|no_tls_err| {
                    anyhow!(
                        "tls connect error: {}; non-tls connect error: {}",
                        tls_err,
                        no_tls_err
                    )
                })
            }
        }
    }
}

fn apply_analyze_profile(mut args: AnalyzeArgs) -> Result<AnalyzeArgs> {
    let Some(profile_raw) = args.profile.clone() else {
        return Ok(args);
    };
    let profile = profile_raw.trim().to_string();

    if let Some(cfg_path) = args.profile_config.clone() {
        let text = fs::read_to_string(&cfg_path).map_err(|e| {
            anyhow!(
                "failed to read profile config '{}': {}",
                cfg_path.display(),
                e
            )
        })?;
        let cfg: ProfileConfigFile = toml::from_str(&text).map_err(|e| {
            anyhow!(
                "failed to parse profile config '{}': {}",
                cfg_path.display(),
                e
            )
        })?;
        let entry = cfg
            .profiles
            .get(&profile)
            .or_else(|| {
                cfg.profiles
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(&profile))
                    .map(|(_, v)| v)
            })
            .ok_or_else(|| {
                anyhow!(
                    "profile '{}' not found in {}",
                    profile_raw,
                    cfg_path.display()
                )
            })?;
        apply_profile_entry(&mut args, entry);
        return Ok(args);
    }

    // Built-in profiles are generic only (no dataset-specific column names).
    match profile.to_lowercase().as_str() {
        "exec" => {
            if args.auto_group_k == 5 {
                args.auto_group_k = 3;
            }
            if args.top == 20 {
                args.top = 12;
            }
            if args.min_records == 1 {
                args.min_records = 20;
            }
        }
        "segment" => {
            if args.auto_group_k == 5 {
                args.auto_group_k = 5;
            }
            if args.top == 20 {
                args.top = 20;
            }
            if args.min_records == 1 {
                args.min_records = 20;
            }
        }
        "supplier" => {
            if args.auto_group_k == 5 {
                args.auto_group_k = 3;
            }
            if args.top == 20 {
                args.top = 20;
            }
            if args.min_records == 1 {
                args.min_records = 10;
            }
        }
        _ => {
            return Err(anyhow!(
                "unknown profile '{}'. Built-ins: exec, segment, supplier. Or pass --profile-config <path.toml>.",
                profile_raw
            ));
        }
    }

    Ok(args)
}

fn apply_profile_entry(args: &mut AnalyzeArgs, entry: &AnalyzeProfile) {
    if args.group_by.is_empty() {
        if let Some(v) = &entry.group_by {
            args.group_by = v.clone();
        }
    }
    if args.metrics.is_empty() {
        if let Some(v) = &entry.metrics {
            args.metrics = v.clone();
        }
    }
    if args.r#where.is_empty() {
        if let Some(v) = &entry.where_clauses {
            args.r#where = v.clone();
        }
    }
    if args.rank_by.is_none() {
        if let Some(v) = &entry.rank_by {
            args.rank_by = Some(v.clone());
        }
    }
    if args.top == 20 {
        if let Some(v) = entry.top {
            args.top = v;
        }
    }
    if args.min_records == 1 {
        if let Some(v) = entry.min_records {
            args.min_records = v;
        }
    }
    if args.auto_group_k == 5 {
        if let Some(v) = entry.auto_group_k {
            args.auto_group_k = v;
        }
    }
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
        Some(1) => "Primary Shared Move",
        Some(2) => "Secondary Shared Move",
        Some(3) => "Cross-Asset Spread",
        Some(4) => "Higher-Order Pattern",
        Some(5) => "Higher-Order Pattern",
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

fn auto_factor_labels(summary: &factor_io::ArtifactSummary) -> FactorLabels {
    let mut out = HashMap::new();
    let k = summary.model.k;
    let tickers = &summary.model.tickers;
    let loadings = &summary.model.loadings;

    for factor_i in 0..k {
        let mut pairs = tickers
            .iter()
            .enumerate()
            .map(|(asset_i, t)| {
                let v = loadings
                    .get(asset_i)
                    .and_then(|row| row.get(factor_i))
                    .copied()
                    .unwrap_or(0.0);
                (t.clone(), v)
            })
            .collect::<Vec<_>>();

        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let low_list = pairs
            .iter()
            .take(2)
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>();
        let high_list = pairs
            .iter()
            .rev()
            .filter(|(t, _)| !low_list.iter().any(|x| *x == t.as_str()))
            .take(2)
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>();
        let lows = low_list.join("/");
        let highs = if high_list.is_empty() {
            "n/a".to_string()
        } else {
            high_list.join("/")
        };

        let base = factor_normal_name(&format!("factor_{}", factor_i + 1));
        let label = format!("{} (high: {}; low: {})", base, highs, lows);
        out.insert(format!("factor_{}_contrib", factor_i + 1), label);
    }

    out
}

fn distinct_tickers(prices: &[factor_core::PricePoint]) -> Vec<String> {
    let mut set = HashSet::new();
    let mut out = Vec::new();
    for p in prices {
        if set.insert(p.ticker.clone()) {
            out.push(p.ticker.clone());
        }
    }
    out.sort();
    out
}

fn read_factors_csv(path: &PathBuf) -> Result<FactorTable> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    if headers.len() < 2 {
        return Err(anyhow!(
            "factors CSV must contain date plus at least one factor column"
        ));
    }
    let date_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("date"))
        .unwrap_or(0);
    let factor_cols = headers
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != date_idx)
        .map(|(i, h)| (i, h.to_string()))
        .collect::<Vec<_>>();

    let mut dates = Vec::new();
    let mut values = HashMap::new();
    for rec in rdr.records() {
        let rec = rec?;
        let date_str = rec
            .get(date_idx)
            .ok_or_else(|| anyhow!("missing factor date"))?
            .trim();
        let date = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            .map_err(|_| anyhow!("invalid factor date '{}'", date_str))?;
        let row = factor_cols
            .iter()
            .map(|(idx, _)| {
                rec.get(*idx)
                    .unwrap_or("")
                    .trim()
                    .parse::<f64>()
                    .unwrap_or(0.0)
            })
            .collect::<Vec<_>>();
        dates.push(date);
        values.insert(date, row);
    }
    if values.is_empty() {
        return Err(anyhow!("factors CSV has no rows"));
    }

    Ok(FactorTable {
        dates,
        factor_names: factor_cols.into_iter().map(|(_, h)| h).collect(),
        values,
    })
}

fn align_portfolio_and_factors(
    prices: &[factor_core::PricePoint],
    weights: &[PortfolioWeight],
    factors: &FactorTable,
) -> Result<AlignedRegressionData> {
    let mut by_ticker: BTreeMap<String, Vec<(chrono::NaiveDate, f64)>> = BTreeMap::new();
    for p in prices {
        by_ticker
            .entry(p.ticker.clone())
            .or_default()
            .push((p.date, p.close));
    }

    let weight_map = weights
        .iter()
        .map(|w| (w.ticker.clone(), w.weight))
        .collect::<HashMap<_, _>>();
    let mut sum = 0.0;
    for (t, w) in &weight_map {
        if by_ticker.contains_key(t) {
            sum += *w;
        }
    }
    if sum.abs() < 1e-12 {
        return Err(anyhow!(
            "portfolio weights do not overlap with prices tickers"
        ));
    }

    let mut returns_by_ticker: HashMap<String, HashMap<chrono::NaiveDate, f64>> = HashMap::new();
    for (ticker, mut series) in by_ticker {
        series.sort_by_key(|x| x.0);
        let mut map = HashMap::new();
        for w in series.windows(2) {
            let (d0, p0) = w[0];
            let (d1, p1) = w[1];
            if p0 > 0.0 && d1 > d0 {
                map.insert(d1, (p1 / p0).ln());
            }
        }
        returns_by_ticker.insert(ticker, map);
    }

    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut dates = Vec::new();
    for d in &factors.dates {
        let mut pr = 0.0;
        let mut has_any = false;
        for (ticker, w) in &weight_map {
            if let Some(ret_map) = returns_by_ticker.get(ticker) {
                if let Some(r) = ret_map.get(d) {
                    pr += (w / sum) * r;
                    has_any = true;
                }
            }
        }
        if !has_any {
            continue;
        }
        let fv = factors
            .values
            .get(d)
            .ok_or_else(|| anyhow!("missing factor row for {}", d))?;
        dates.push(*d);
        y.push(pr);
        x.push(fv.clone());
    }
    if dates.len() < factors.factor_names.len() + 5 {
        return Err(anyhow!(
            "not enough aligned observations for regression (have {}, need more than factors+4)",
            dates.len()
        ));
    }

    Ok(AlignedRegressionData {
        dates,
        y,
        x,
        factor_names: factors.factor_names.clone(),
    })
}

fn ols_regression(
    dates: &[chrono::NaiveDate],
    y: &[f64],
    x: &[Vec<f64>],
    factor_names: &[String],
) -> Result<RegressionResult> {
    let n = y.len();
    let p = factor_names.len();
    let mut data = Vec::with_capacity(n * (p + 1));
    for row in x {
        data.push(1.0);
        data.extend(row.iter().copied());
    }

    let xmat = DMatrix::from_row_slice(n, p + 1, &data);
    let yvec = DVector::from_row_slice(y);
    let xtx = xmat.transpose() * &xmat;
    let xty = xmat.transpose() * &yvec;
    let beta = xtx
        .clone()
        .lu()
        .solve(&xty)
        .ok_or_else(|| anyhow!("regression solve failed (singular matrix)"))?;

    let fitted = xmat * &beta;
    let residuals = yvec.clone() - &fitted;

    let y_mean = y.iter().sum::<f64>() / n as f64;
    let sst = y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>();
    let sse = residuals.iter().map(|e| e.powi(2)).sum::<f64>();
    let r2 = if sst.abs() < 1e-12 {
        0.0
    } else {
        1.0 - (sse / sst)
    };
    let dof = (n as f64 - (p as f64 + 1.0)).max(1.0);
    let residual_std = (sse / dof).sqrt();

    Ok(RegressionResult {
        dates: dates.to_vec(),
        factor_names: factor_names.to_vec(),
        alpha: beta[0],
        betas: beta.iter().skip(1).copied().collect(),
        r2,
        residual_std,
        observations: n,
        residuals: residuals.iter().copied().collect(),
        fitted: fitted.iter().copied().collect(),
        y: y.to_vec(),
    })
}

fn write_regression_artifacts(out_dir: &PathBuf, reg: &RegressionResult) -> Result<()> {
    fs::create_dir_all(out_dir)?;

    let mut beta_map = serde_json::Map::new();
    for (name, beta) in reg.factor_names.iter().zip(reg.betas.iter()) {
        beta_map.insert(name.clone(), serde_json::json!(beta));
    }
    let json = serde_json::json!({
        "method": "ols_known_factor_regression",
        "observations": reg.observations,
        "alpha": reg.alpha,
        "betas": beta_map,
        "r2": reg.r2,
        "residual_std": reg.residual_std
    });
    fs::write(
        out_dir.join("regression.json"),
        serde_json::to_string_pretty(&json)?,
    )?;

    let mut wtr = csv::Writer::from_path(out_dir.join("regression_residuals.csv"))?;
    wtr.write_record(["date", "portfolio_return", "fitted_return", "residual"])?;
    for i in 0..reg.observations {
        let date = reg
            .dates
            .get(i)
            .map(|d| d.to_string())
            .unwrap_or_else(|| "".to_string());
        wtr.write_record([
            date,
            format!("{:.10}", reg.y[i]),
            format!("{:.10}", reg.fitted[i]),
            format!("{:.10}", reg.residuals[i]),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

struct AnalysisReport {
    markdown: String,
    json: serde_json::Value,
    used_groups: Vec<String>,
}

#[derive(Debug, Clone)]
struct PeriodCompareConfig {
    date_column: String,
    time_grain: Option<TimeGrain>,
    period: Option<PeriodPreset>,
    current_start: chrono::NaiveDate,
    current_end: chrono::NaiveDate,
    previous_start: chrono::NaiveDate,
    previous_end: chrono::NaiveDate,
}

struct FactorTable {
    dates: Vec<chrono::NaiveDate>,
    factor_names: Vec<String>,
    values: HashMap<chrono::NaiveDate, Vec<f64>>,
}

struct AlignedRegressionData {
    dates: Vec<chrono::NaiveDate>,
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    factor_names: Vec<String>,
}

#[derive(Debug)]
struct RegressionResult {
    dates: Vec<chrono::NaiveDate>,
    factor_names: Vec<String>,
    alpha: f64,
    betas: Vec<f64>,
    r2: f64,
    residual_std: f64,
    observations: usize,
    residuals: Vec<f64>,
    fitted: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct ProfileConfigFile {
    profiles: HashMap<String, AnalyzeProfile>,
}

#[derive(Debug, Deserialize, Clone)]
struct AnalyzeProfile {
    group_by: Option<Vec<String>>,
    metrics: Option<Vec<String>>,
    where_clauses: Option<Vec<String>>,
    rank_by: Option<String>,
    top: Option<usize>,
    min_records: Option<u64>,
    auto_group_k: Option<usize>,
}

fn analyze_table_csv(
    input: &PathBuf,
    profile: Option<&str>,
    group_by: &[String],
    auto_group_k: usize,
    metrics: &[String],
    count_only: bool,
    agg: AggKind,
    percentiles: &[PercentileKind],
    normalize_text_groups: bool,
    word_freq: bool,
    period_cfg: Option<&PeriodCompareConfig>,
    where_clauses: &[String],
    exclude_blank_groups: bool,
    rank_by: Option<&str>,
    top_n: usize,
    top_insights: usize,
    opportunity_min_records: u64,
    min_records: u64,
    alert_top5_share: Option<f64>,
    alert_blank_share: Option<f64>,
    alert_rules: &[String],
) -> Result<AnalysisReport> {
    let mut rdr = csv::Reader::from_path(input)?;
    let headers = rdr.headers()?.clone();

    let resolved_groups = if group_by.is_empty() {
        auto_detect_groups(&headers, input, auto_group_k)?
    } else {
        group_by
            .iter()
            .map(|g| resolve_group_name(g, &headers))
            .collect::<Result<Vec<_>>>()?
    };
    if resolved_groups.is_empty() {
        return Err(anyhow!("no grouping columns selected or detected"));
    }

    let metric_cols = if count_only {
        Vec::new()
    } else if metrics.is_empty() {
        auto_detect_numeric_metrics(input, &headers, &resolved_groups, 3)?
    } else {
        let cols = metrics
            .iter()
            .map(|m| {
                headers
                    .iter()
                    .position(|h| h == m)
                    .map(|idx| (m.to_string(), idx))
                    .ok_or_else(|| anyhow!("metric column '{}' not found", m))
            })
            .collect::<Result<Vec<_>>>()?;
        cols
    };
    if count_only {
        if let Some(rb) = rank_by {
            if rb != "count" {
                return Err(anyhow!(
                    "--count-only supports ranking by count only; remove --rank-by or use --rank-by count"
                ));
            }
        }
    }
    let rank_metric = if count_only {
        None
    } else {
        rank_by.and_then(|rb| {
            metric_cols
                .iter()
                .find(|(m, _)| m == rb)
                .map(|(m, _)| m.clone())
        })
    };
    if !count_only {
        if let Some(rb) = rank_by {
        if rank_metric.is_none() {
            return Err(anyhow!(
                "rank metric '{}' not found. Available metrics: {}",
                rb,
                metric_cols
                    .iter()
                    .map(|(m, _)| m.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    }

    let group_idxs = resolved_groups
        .iter()
        .map(|g| {
            headers
                .iter()
                .position(|h| h == g)
                .ok_or_else(|| anyhow!("group column not found: {}", g))
        })
        .collect::<Result<Vec<_>>>()?;
    let date_idx = if let Some(cfg) = period_cfg {
        Some(
            headers
                .iter()
                .position(|h| h == cfg.date_column)
                .or_else(|| {
                    headers
                        .iter()
                        .position(|h| h.eq_ignore_ascii_case(&cfg.date_column))
                })
                .ok_or_else(|| anyhow!("date column '{}' not found", cfg.date_column))?,
        )
    } else {
        None
    };
    let where_filters = parse_where_filters(where_clauses, &headers)?;
    let word_group_cols = resolved_groups
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            if should_normalize_group_column(name) {
                Some((i, name.clone()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let (period_group_idxs, period_group_names) = if let Some(didx) = date_idx {
        let filtered = group_idxs
            .iter()
            .enumerate()
            .filter_map(|(i, idx)| {
                if *idx == didx {
                    None
                } else {
                    Some((*idx, resolved_groups[i].clone()))
                }
            })
            .collect::<Vec<_>>();
        if filtered.is_empty() {
            (group_idxs.clone(), resolved_groups.clone())
        } else {
            let idxs = filtered.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
            let names = filtered
                .into_iter()
                .map(|(_, name)| name)
                .collect::<Vec<_>>();
            (idxs, names)
        }
    } else {
        (group_idxs.clone(), resolved_groups.clone())
    };

    let mut by_group: BTreeMap<String, (u64, HashMap<String, Vec<f64>>)> = BTreeMap::new();
    let mut word_counts: HashMap<String, u64> = HashMap::new();
    let mut row_count = 0_u64;
    let primary_metric = metric_cols.first().map(|(m, _)| m.clone());
    let primary_idx = primary_metric.as_ref().and_then(|pm| {
        metric_cols
            .iter()
            .find(|(m, _)| m == pm)
            .map(|(_, idx)| *idx)
    });
    let mut period_by_group: HashMap<String, (u64, f64, u64, f64)> = HashMap::new();
    let mut period_totals = (0_u64, 0.0_f64, 0_u64, 0.0_f64);

    for rec in rdr.records() {
        let rec = rec?;
        if !matches_where_filters(&rec, &where_filters) {
            continue;
        }
        let gk = compose_group_key(
            &rec,
            &group_idxs,
            &resolved_groups,
            normalize_text_groups,
        );
        if exclude_blank_groups && is_blank_group_key(&gk) {
            continue;
        }
        row_count += 1;

        if word_freq {
            for (i, _) in &word_group_cols {
                let raw = rec.get(group_idxs[*i]).unwrap_or("").trim();
                for w in tokenize_words(raw) {
                    *word_counts.entry(w).or_insert(0) += 1;
                }
            }
        }

        let entry = by_group
            .entry(gk.clone())
            .or_insert_with(|| (0, HashMap::<String, Vec<f64>>::new()));
        entry.0 += 1;

        for (name, idx) in &metric_cols {
            let raw = rec.get(*idx).unwrap_or("").trim();
            if let Some(v) = parse_numeric(raw) {
                entry.1.entry(name.clone()).or_default().push(v);
            }
        }

        if let (Some(cfg), Some(didx)) = (period_cfg, date_idx) {
            if let Some(d) = parse_date_like(rec.get(didx).unwrap_or("").trim()) {
                let gk_period = compose_group_key(
                    &rec,
                    &period_group_idxs,
                    &period_group_names,
                    normalize_text_groups,
                );
                let primary_val = primary_idx
                    .and_then(|pidx| parse_numeric(rec.get(pidx).unwrap_or("").trim()))
                    .unwrap_or(0.0);
                if d >= cfg.current_start && d <= cfg.current_end {
                    let e = period_by_group
                        .entry(gk_period.clone())
                        .or_insert((0, 0.0, 0, 0.0));
                    e.0 += 1;
                    e.1 += primary_val;
                    period_totals.0 += 1;
                    period_totals.1 += primary_val;
                } else if d >= cfg.previous_start && d <= cfg.previous_end {
                    let e = period_by_group
                        .entry(gk_period.clone())
                        .or_insert((0, 0.0, 0, 0.0));
                    e.2 += 1;
                    e.3 += primary_val;
                    period_totals.2 += 1;
                    period_totals.3 += primary_val;
                }
            }
        }

    }
    let total_count_all = by_group.values().map(|(c, _)| *c).sum::<u64>();
    let blank_count_all = by_group
        .iter()
        .filter(|(g, _)| is_blank_group_key(g))
        .map(|(_, (c, _))| *c)
        .sum::<u64>();
    let total_primary_all = if let Some(pm) = &primary_metric {
        by_group
            .values()
            .map(|(_, vals)| {
                vals.get(pm)
                    .map(|xs| aggregate_values(xs, agg))
                    .unwrap_or(0.0)
            })
            .sum::<f64>()
    } else {
        0.0
    };

    let mut rows = by_group
        .into_iter()
        .map(|(group, (count, values))| {
            let mut aggregates = HashMap::new();
            for (name, xs) in values {
                aggregates.insert(name.clone(), aggregate_values(&xs, agg));
                for pct in percentiles {
                    aggregates.insert(
                        format!("{}_{}", name, pct.label()),
                        percentile_value(&xs, pct.quantile()),
                    );
                }
            }
            (group, count, aggregates)
        })
        .filter(|(_, count, _)| *count >= min_records)
        .collect::<Vec<_>>();
    if let Some(rm) = &rank_metric {
        rows.sort_by(|a, b| {
            let av = a.2.get(rm).copied().unwrap_or(0.0);
            let bv = b.2.get(rm).copied().unwrap_or(0.0);
            bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        rows.sort_by(|a, b| b.1.cmp(&a.1));
    }
    let segment_count = rows.len();

    let mut md = String::new();
    md.push_str("# Analysis Brief\n\n");
    md.push_str(&format!("- Input: {}\n", input.display()));
    md.push_str(&format!("- Records (after filters): {}\n", row_count));
    md.push_str(&format!(
        "- Segments (distinct grouped combinations): {}\n",
        segment_count
    ));
    md.push_str(&format!("- Grouped by: {}\n", resolved_groups.join(", ")));
    if let Some(p) = profile {
        md.push_str(&format!("- Profile: {}\n", p));
    }
    if where_filters.is_empty() {
        md.push_str("- Filters: none\n");
    } else {
        md.push_str(&format!(
            "- Filters: {}\n",
            where_filters
                .iter()
                .map(|(name, _, val)| format!("{}={}", name, val))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    md.push_str(&format!(
        "- Blank groups: {}\n",
        if exclude_blank_groups {
            "excluded"
        } else {
            "included"
        }
    ));
    md.push_str(&format!(
        "- Ranking: {}\n",
        rank_metric.clone().unwrap_or_else(|| "count".to_string())
    ));
    md.push_str(&format!("- Top rows shown: {}\n", top_n));
    if top_insights > 0 {
        md.push_str(&format!("- Top insights requested: {}\n", top_insights));
        md.push_str(&format!(
            "- Opportunity min records: {}\n",
            opportunity_min_records
        ));
    }
    md.push_str(&format!("- Minimum records per segment: {}\n", min_records));
    md.push_str(&format!(
        "- Metric aggregation: {}\n",
        if count_only { "Count-only (no numeric metrics)" } else { agg.label() }
    ));
    if !percentiles.is_empty() {
        md.push_str(&format!(
            "- Extra percentile columns: {}\n",
            percentiles
                .iter()
                .map(|p| p.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if normalize_text_groups {
        md.push_str("- Text normalization for name/title groups: enabled\n");
    }
    if metric_cols.is_empty() {
        if count_only {
            md.push_str("- Metrics: disabled via --count-only\n\n");
        } else {
            md.push_str("- Metrics: none detected (count-only analysis)\n\n");
        }
    } else {
        md.push_str(&format!(
            "- Metrics: {}\n\n",
            metric_cols
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let total_primary = total_primary_all;
    let total_count = total_count_all;
    let top1 = rows
        .iter()
        .filter(|(group, _, _)| !is_blank_group_key(group))
        .max_by_key(|(_, count, _)| *count)
        .or_else(|| rows.first());
    let top5_count = rows.iter().take(5).map(|(_, c, _)| *c).sum::<u64>();
    let top5_names = rows
        .iter()
        .take(5)
        .map(|(g, _, _)| format!("`{}`", g))
        .collect::<Vec<_>>();
    let top5_primary = if let Some(pm) = &primary_metric {
        rows.iter()
            .take(5)
            .map(|(_, _, sums)| sums.get(pm).copied().unwrap_or(0.0))
            .sum::<f64>()
    } else {
        0.0
    };
    let top5_n = rows.iter().take(5).count() as f64;
    let top5_primary_avg = if top5_n > 0.0 {
        top5_primary / top5_n
    } else {
        0.0
    };
    let top5_count_pct = if total_count > 0 {
        (top5_count as f64 / total_count as f64) * 100.0
    } else {
        0.0
    };
    let blank_share_pct = if total_count > 0 {
        (blank_count_all as f64 / total_count as f64) * 100.0
    } else {
        0.0
    };

    let mut alerts = Vec::new();
    if let Some(threshold) = alert_top5_share {
        if top5_count_pct >= threshold {
            alerts.push(format!(
                "High concentration: top 5 segments account for {:.1}% of records (threshold {:.1}%).",
                top5_count_pct, threshold
            ));
        }
    }
    if let Some(threshold) = alert_blank_share {
        if blank_share_pct >= threshold {
            alerts.push(format!(
                "High blank-segment share: {:.1}% of records are grouped as (blank) (threshold {:.1}%).",
                blank_share_pct, threshold
            ));
        }
    }

    let mut alert_rule_results = Vec::new();
    if !alert_rules.is_empty() {
        let ctx = AlertEvalContext {
            top5_record_share_pct: top5_count_pct,
            blank_share_pct,
            segments: segment_count as f64,
            records: total_count as f64,
        };
        for raw in alert_rules {
            let parsed = parse_alert_rule(raw)?;
            let matched = eval_alert_rule(&parsed, &ctx);
            if matched {
                alerts.push(format!(
                    "Rule triggered: {} (actual={:.3})",
                    raw.trim(),
                    alert_metric_value(&parsed.metric, &ctx)
                ));
            }
            alert_rule_results.push(serde_json::json!({
                "rule": raw.trim(),
                "metric": parsed.metric,
                "operator": parsed.op,
                "threshold": parsed.threshold,
                "actual": alert_metric_value(&parsed.metric, &ctx),
                "triggered": matched
            }));
        }
    }

    md.push_str("## Executive Summary\n\n");
    if let Some((group, count, sums)) = top1 {
        let count_pct = if total_count > 0 {
            (*count as f64 / total_count as f64) * 100.0
        } else {
            0.0
        };
        let primary_line = if let Some(pm) = &primary_metric {
            let v = sums.get(pm).copied().unwrap_or(0.0);
            let share = if agg == AggKind::Sum && total_primary.abs() > 1e-12 {
                (v / total_primary) * 100.0
            } else {
                0.0
            };
            if agg == AggKind::Sum {
                format!(" and {:.1}% of total {}", share, pm)
            } else {
                format!(" and {} {} ({})", fmt_num(v, 2), pm, agg.label())
            }
        } else {
            String::new()
        };
        md.push_str(&format!(
            "- Largest segment is `{}` with {:.1}% of records{}.\n",
            group, count_pct, primary_line
        ));
    }
    md.push_str(&format!(
        "- Top 5 segments represent {:.1}% of records",
        top5_count_pct
    ));
    if let Some(pm) = &primary_metric {
        let pct = if total_primary.abs() > 1e-12 {
            (top5_primary / total_primary) * 100.0
        } else {
            0.0
        };
        if agg == AggKind::Sum {
            md.push_str(&format!(" and {:.1}% of {}.\n", pct, pm));
        } else {
            md.push_str(&format!(
                " and average {} {} across top 5 segments ({}).\n",
                fmt_num(top5_primary_avg, 2),
                pm,
                agg.label()
            ));
        }
    } else {
        md.push_str(".\n");
    }
    if !top5_names.is_empty() {
        md.push_str(&format!(
            "- Top 5 segment names: {}.\n",
            top5_names.join(", ")
        ));
    }
    md.push('\n');

    if let Some(cfg) = period_cfg {
        let current_total = period_totals.1;
        let previous_total = period_totals.3;
        let total_delta = current_total - previous_total;
        let total_delta_pct = if previous_total.abs() > 1e-12 {
            (total_delta / previous_total.abs()) * 100.0
        } else {
            0.0
        };
        let mut movers = period_by_group
            .iter()
            .map(|(g, (cc, cv, pc, pv))| {
                let d_metric = cv - pv;
                let d_count = *cc as i64 - *pc as i64;
                (g.clone(), *cc, *cv, *pc, *pv, d_metric, d_count)
            })
            .collect::<Vec<_>>();
        movers.sort_by(|a, b| {
            a.5.partial_cmp(&b.5)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut concentration_movers = period_by_group
            .iter()
            .map(|(g, (cc, cv, pc, pv))| {
                let base_share = pct(*pc, period_totals.2);
                let new_share = pct(*cc, period_totals.0);
                let d_share = new_share - base_share;
                (g.clone(), *pc, *cc, base_share, new_share, d_share, *pv, *cv)
            })
            .collect::<Vec<_>>();
        concentration_movers.sort_by(|a, b| {
            b.5.abs()
                .partial_cmp(&a.5.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let prev_seg_count = period_by_group.values().filter(|(_, _, pc, _)| *pc > 0).count();
        let curr_seg_count = period_by_group.values().filter(|(cc, _, _, _)| *cc > 0).count();
        let prev_top5_count = concentration_movers.iter().take(5).map(|x| x.1).sum::<u64>();
        let curr_top5_count = concentration_movers.iter().take(5).map(|x| x.2).sum::<u64>();
        let prev_top5_pct = pct(prev_top5_count, period_totals.2);
        let curr_top5_pct = pct(curr_top5_count, period_totals.0);

        md.push_str("## Period Comparison\n\n");
        md.push_str(&format!(
            "- Date column: `{}`\n",
            cfg.date_column
        ));
        if let (Some(g), Some(p)) = (cfg.time_grain, cfg.period) {
            md.push_str(&format!(
                "- Window mode: {:?} / {:?}\n",
                g, p
            ));
        }
        md.push_str(&format!(
            "- Current window: {} to {} (records={})\n",
            cfg.current_start, cfg.current_end, period_totals.0
        ));
        md.push_str(&format!(
            "- Previous window: {} to {} (records={})\n",
            cfg.previous_start, cfg.previous_end, period_totals.2
        ));
        md.push('\n');
        md.push_str("### Executive Delta\n\n");
        md.push_str(&format!(
            "- Top-5 concentration changed from {:.1}% to {:.1}% ({:+.1} pp).\n",
            prev_top5_pct,
            curr_top5_pct,
            curr_top5_pct - prev_top5_pct
        ));
        md.push_str(&format!(
            "- Segment count changed from {} to {} ({:+}).\n",
            prev_seg_count,
            curr_seg_count,
            curr_seg_count as i64 - prev_seg_count as i64
        ));
        md.push('\n');
        md.push_str("### Top Concentration Changes\n\n");
        for (i, (g, pc, cc, _bs, _ns, d_share, _pv, _cv)) in concentration_movers.iter().take(5).enumerate() {
            md.push_str(&format!(
                "{}. `{}`   {} -> {} records ({:+.1} pp)\n",
                i + 1,
                g,
                pc,
                cc,
                d_share
            ));
        }
        md.push('\n');
        if let Some(pm) = &primary_metric {
            let arrow = if total_delta < 0.0 { "↓" } else { "↑" };
            md.push_str(&format!(
                "- {} {} {:.1}% ({} -> {}, delta={}).\n",
                pm,
                arrow,
                total_delta_pct.abs(),
                fmt_num(previous_total, 2),
                fmt_num(current_total, 2),
                fmt_num(total_delta, 2)
            ));
            md.push_str(&format!(
                "- Group drivers (largest declines) by `{}`:\n",
                period_group_names.join(", ")
            ));
            for (g, _cc, cv, _pc, pv, dm, _dc) in movers.iter().take(5) {
                let pct = if pv.abs() > 1e-12 {
                    (dm / pv.abs()) * 100.0
                } else {
                    0.0
                };
                md.push_str(&format!(
                    "  - `{}`: {} -> {} (delta={}, {:+.1}%)\n",
                    g,
                    fmt_num(*pv, 2),
                    fmt_num(*cv, 2),
                    fmt_num(*dm, 2),
                    pct
                ));
            }
        } else {
            let count_delta = period_totals.0 as i64 - period_totals.2 as i64;
            md.push_str(&format!(
                "- Record delta: {} (current={} vs previous={}).\n",
                count_delta, period_totals.0, period_totals.2
            ));
        }
        md.push('\n');
    }

    if !alerts.is_empty() {
        md.push_str("## Alerts\n\n");
        for a in &alerts {
            md.push_str(&format!("- {}\n", a));
        }
        md.push('\n');
    }

    let top_words = if word_freq {
        top_word_counts(&word_counts, 12)
    } else {
        Vec::new()
    };
    if !top_words.is_empty() {
        md.push_str("## Top Words\n\n");
        for (w, c) in &top_words {
            md.push_str(&format!("- `{}`: {}\n", w, c));
        }
        md.push('\n');
    }

    md.push_str("## Insights\n\n");
    if let Some((group, count, sums)) = top1 {
        let pct = if total_count > 0 {
            (*count as f64 / total_count as f64) * 100.0
        } else {
            0.0
        };
        md.push_str(&format!(
            "- Top segment by count: `{}` with {} records ({:.1}% of all records).\n",
            group, count, pct
        ));
        if let Some(pm) = &primary_metric {
            let top_val = sums.get(pm).copied().unwrap_or(0.0);
            let pm_pct = if agg == AggKind::Sum && total_primary.abs() > 1e-12 {
                (top_val / total_primary) * 100.0
            } else {
                0.0
            };
            if agg == AggKind::Sum {
                md.push_str(&format!(
                    "- Top segment contributes {} `{}` ({:.1}% of total `{}`).\n",
                    fmt_num(top_val, 2),
                    pm,
                    pm_pct,
                    pm
                ));
            } else {
                md.push_str(&format!(
                    "- Top segment {} `{}` is {}.\n",
                    agg.label().to_lowercase(),
                    pm,
                    fmt_num(top_val, 2)
                ));
            }
        }
    }
    md.push_str(&format!(
        "- Concentration: top 5 segments account for {} records ({:.1}% of all records).\n",
        top5_count, top5_count_pct
    ));
    if let Some(pm) = &primary_metric {
        if agg == AggKind::Sum {
            let pct = if total_primary.abs() > 1e-12 {
                (top5_primary / total_primary) * 100.0
            } else {
                0.0
            };
            md.push_str(&format!(
                "- Concentration by `{}`: top 5 segments represent {} ({:.1}% of total).\n",
                pm,
                fmt_num(top5_primary, 2),
                pct
            ));
        } else {
            md.push_str(&format!(
                "- Top 5 segments average `{}` as {} ({}).\n",
                pm,
                fmt_num(top5_primary_avg, 2),
                agg.label()
            ));
        }
    }

    md.push('\n');

    let mut top_risks = Vec::<String>::new();
    let mut top_opportunities = Vec::<String>::new();
    if top_insights > 0 {
        let mut by_count_share = rows
            .iter()
            .map(|(group, count, sums)| {
                let count_share = if total_count > 0 {
                    (*count as f64 / total_count as f64) * 100.0
                } else {
                    0.0
                };
                let primary_value = if let Some(pm) = &primary_metric {
                    sums.get(pm).copied().unwrap_or(0.0)
                } else {
                    0.0
                };
                let primary_share = if agg == AggKind::Sum && total_primary.abs() > 1e-12 {
                    (primary_value / total_primary) * 100.0
                } else {
                    0.0
                };
                let per_record = if *count > 0 {
                    primary_value / *count as f64
                } else {
                    0.0
                };
                (
                    group.clone(),
                    *count,
                    count_share,
                    primary_value,
                    primary_share,
                    per_record,
                )
            })
            .collect::<Vec<_>>();

        by_count_share.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (group, count, count_share, _, primary_share, _) in
            by_count_share.iter().take(top_insights)
        {
            if let Some(pm) = &primary_metric {
                if agg == AggKind::Sum {
                    top_risks.push(format!(
                        "`{}` concentration: {} records ({:.1}% of total records), {:.1}% of total {}.",
                        group, count, count_share, primary_share, pm
                    ));
                } else {
                    top_risks.push(format!(
                        "`{}` concentration: {} records ({:.1}% of total records).",
                        group, count, count_share
                    ));
                }
            } else {
                top_risks.push(format!(
                    "`{}` concentration: {} records ({:.1}% of total records).",
                    group, count, count_share
                ));
            }
        }

        if let Some(pm) = &primary_metric {
            let mut by_per_record = by_count_share.clone();
            by_per_record.sort_by(|a, b| {
                b.5.partial_cmp(&a.5)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (group, count, _, _, _, per_record) in by_per_record
                .iter()
                .filter(|(g, c, _, _, _, _)| {
                    !is_blank_group_key(g) && *c >= opportunity_min_records
                })
                .take(top_insights)
            {
                top_opportunities.push(format!(
                    "`{}` has high {} per record ({}) across {} records.",
                    group,
                    pm,
                    fmt_num(*per_record, 2),
                    count
                ));
            }
        } else {
            if top5_count_pct < 40.0 {
                top_opportunities.push(format!(
                    "Low concentration tail opportunity: top 5 segments are {:.1}% of records.",
                    top5_count_pct
                ));
            }
            if !top5_names.is_empty() {
                top_opportunities.push(format!(
                    "Prioritize the top segments first: {}.",
                    top5_names
                        .iter()
                        .take(top_insights)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        md.push_str("## Top Insights\n\n");
        md.push_str("### Risks\n\n");
        for line in &top_risks {
            md.push_str(&format!("- {}\n", line));
        }
        md.push('\n');
        md.push_str("### Opportunities\n\n");
        for line in &top_opportunities {
            md.push_str(&format!("- {}\n", line));
        }
        md.push('\n');
    }

    let mut display_metrics = metric_cols
        .iter()
        .map(|(m, _)| m.clone())
        .collect::<Vec<_>>();
    for (m, _) in &metric_cols {
        for pct in percentiles {
            display_metrics.push(format!("{}_{}", m, pct.label()));
        }
    }

    md.push_str("## Top Groups\n\n");
    md.push_str("|");
    for g in &resolved_groups {
        md.push_str(&format!(" {} |", g));
    }
    md.push_str(" Records | Record Share |");
    for m in &display_metrics {
        if agg == AggKind::Sum && m == primary_metric.as_deref().unwrap_or("") {
            md.push_str(&format!(" {} | {} Share |", m, m));
        } else {
            md.push_str(&format!(" {} |", m));
        }
    }
    md.push('\n');
    md.push_str("|");
    for _ in &resolved_groups {
        md.push_str("---|");
    }
    md.push_str("---:|---:|");
    for m in &display_metrics {
        if agg == AggKind::Sum && m == primary_metric.as_deref().unwrap_or("") {
            md.push_str("---:|---:|");
        } else {
            md.push_str("---:|");
        }
    }
    md.push('\n');

    for (group, count, sums) in rows.iter().take(top_n) {
        let count_share = if total_count > 0 {
            (*count as f64 / total_count as f64) * 100.0
        } else {
            0.0
        };
        let parts = group
            .split(" | ")
            .map(|x| x.trim().replace('|', "\\|"))
            .collect::<Vec<_>>();
        let mut line = String::from("|");
        for i in 0..resolved_groups.len() {
            let v = parts.get(i).cloned().unwrap_or_default();
            line.push_str(&format!(" {} |", v));
        }
        line.push_str(&format!(" {} | {:.1}% |", count, count_share));
        for m in &display_metrics {
            let v = sums.get(m).copied().unwrap_or(0.0);
            let share = if agg == AggKind::Sum
                && m == primary_metric.as_deref().unwrap_or("")
                && total_primary.abs() > 1e-12
            {
                    (v / total_primary) * 100.0
                } else {
                    0.0
                };
            if agg == AggKind::Sum && m == primary_metric.as_deref().unwrap_or("") {
                line.push_str(&format!(" {} | {:.1}% |", fmt_num(v, 2), share));
            } else {
                line.push_str(&format!(" {} |", fmt_num(v, 2)));
            }
        }
        md.push_str(&line);
        md.push('\n');
    }

    let json_rows = rows
        .into_iter()
        .map(|(group, count, sums)| {
            let mut obj = serde_json::Map::new();
            obj.insert("group".to_string(), serde_json::Value::String(group));
            obj.insert("count".to_string(), serde_json::Value::from(count));
            let count_share = if total_count > 0 {
                (count as f64 / total_count as f64) * 100.0
            } else {
                0.0
            };
            obj.insert(
                "count_share_pct".to_string(),
                serde_json::Value::from(count_share),
            );
            for (k, v) in sums {
                obj.insert(k, serde_json::Value::from(v));
            }
            serde_json::Value::Object(obj)
        })
        .collect::<Vec<_>>();

    let json = serde_json::json!({
        "input": input.display().to_string(),
        "rows": row_count,
        "records": row_count,
        "segments": segment_count,
        "group_by": resolved_groups,
        "filters": where_filters
            .iter()
            .map(|(name, _, val)| format!("{}={}", name, val))
            .collect::<Vec<_>>(),
        "exclude_blank_groups": exclude_blank_groups,
        "metrics": display_metrics,
        "base_metrics": metric_cols.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        "metric_aggregation": agg.label().to_lowercase(),
        "count_only": count_only,
        "percentiles": percentiles.iter().map(|p| p.label()).collect::<Vec<_>>(),
        "normalize_text_groups": normalize_text_groups,
        "word_frequency": top_words
            .iter()
            .map(|(w, c)| serde_json::json!({"word": w, "count": c}))
            .collect::<Vec<_>>(),
        "rank_by": rank_metric.clone().unwrap_or_else(|| "count".to_string()),
        "top": top_n,
        "top_insights": top_insights,
        "opportunity_min_records": opportunity_min_records,
        "min_records": min_records,
        "blank_share_pct": blank_share_pct,
        "alert_thresholds": {
            "top5_share": alert_top5_share,
            "blank_share": alert_blank_share
        },
        "alerts": alerts,
        "alert_rule_results": alert_rule_results,
        "primary_metric": primary_metric,
        "period_compare": period_cfg.map(|cfg| {
            let current_total = period_totals.1;
            let previous_total = period_totals.3;
            let total_delta = current_total - previous_total;
            let total_delta_pct = if previous_total.abs() > 1e-12 {
                (total_delta / previous_total.abs()) * 100.0
            } else {
                0.0
            };
            let mut movers = period_by_group
                .iter()
                .map(|(g, (cc, cv, pc, pv))| {
                    serde_json::json!({
                        "group": g,
                        "current_records": cc,
                        "current_primary_metric_value": cv,
                        "previous_records": pc,
                        "previous_primary_metric_value": pv,
                        "delta_primary_metric_value": cv - pv,
                        "delta_records": *cc as i64 - *pc as i64
                    })
                })
                .collect::<Vec<_>>();
            movers.sort_by(|a, b| {
                a["delta_primary_metric_value"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .partial_cmp(&b["delta_primary_metric_value"].as_f64().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            serde_json::json!({
                "date_column": cfg.date_column,
                "time_grain": cfg.time_grain.map(|x| format!("{:?}", x).to_lowercase()),
                "period": cfg.period.map(|x| format!("{:?}", x).to_lowercase()),
                "current_start": cfg.current_start.to_string(),
                "current_end": cfg.current_end.to_string(),
                "previous_start": cfg.previous_start.to_string(),
                "previous_end": cfg.previous_end.to_string(),
                "current_records": period_totals.0,
                "previous_records": period_totals.2,
                "current_primary_metric_value": current_total,
                "previous_primary_metric_value": previous_total,
                "delta_primary_metric_value": total_delta,
                "delta_primary_metric_pct": total_delta_pct,
                "movers": movers.into_iter().take(20).collect::<Vec<_>>()
            })
        }),
        "top5_count": top5_count,
        "top5_primary_metric_value": top5_primary,
        "top_risks": top_risks,
        "top_opportunities": top_opportunities,
        "groups": json_rows,
    });

    let used_groups = json["group_by"]
        .as_array()
        .map(|xs| {
            xs.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(AnalysisReport {
        markdown: md,
        json,
        used_groups,
    })
}

fn auto_detect_groups(headers: &StringRecord, input: &PathBuf, k: usize) -> Result<Vec<String>> {
    let mut selected = Vec::new();

    let mut rdr = csv::Reader::from_path(input)?;
    let mut distinct: HashMap<String, HashSet<String>> = HashMap::new();
    let mut non_empty: HashMap<String, usize> = HashMap::new();
    let mut rows = 0usize;
    for rec in rdr.records().take(2000) {
        let rec = rec?;
        rows += 1;
        for (idx, h) in headers.iter().enumerate() {
            let v = rec.get(idx).unwrap_or("").trim();
            if v.is_empty() {
                continue;
            }
            *non_empty.entry(h.to_string()).or_insert(0) += 1;
            if v.parse::<f64>().is_err() {
                distinct
                    .entry(h.to_string())
                    .or_default()
                    .insert(v.to_string());
            }
        }
    }

    let mut candidates = distinct
        .into_iter()
        .filter_map(|(h, set)| {
            let card = set.len();
            let fill = non_empty.get(&h).copied().unwrap_or(0);
            let fill_ratio = if rows == 0 {
                0.0
            } else {
                fill as f64 / rows as f64
            };
            if (2..=60).contains(&card) && fill_ratio > 0.2 && !selected.iter().any(|x| x == &h) {
                Some((h, card))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|a, b| a.1.cmp(&b.1));
    for (h, _) in candidates {
        if selected.len() >= k {
            break;
        }
        selected.push(h);
    }
    Ok(selected)
}

fn resolve_group_name(name: &str, headers: &StringRecord) -> Result<String> {
    let n = name.trim();

    if headers.iter().any(|h| h == n) {
        return Ok(n.to_string());
    }
    if let Some(real) = headers.iter().find(|h| h.eq_ignore_ascii_case(n)) {
        return Ok(real.to_string());
    }
    Err(anyhow!("group column '{}' not found", n))
}

fn auto_detect_numeric_metrics(
    input: &PathBuf,
    headers: &StringRecord,
    group_cols: &[String],
    max_metrics: usize,
) -> Result<Vec<(String, usize)>> {
    let mut out = Vec::new();
    let group_set = group_cols.iter().cloned().collect::<HashSet<_>>();

    let mut rdr = csv::Reader::from_path(input)?;
    let mut seen = 0usize;
    let mut numeric_ok = vec![0usize; headers.len()];
    let mut non_empty = vec![0usize; headers.len()];
    for rec in rdr.records().take(1500) {
        let rec = rec?;
        seen += 1;
        for i in 0..headers.len() {
            let v = rec.get(i).unwrap_or("").trim();
            if v.is_empty() {
                continue;
            }
            non_empty[i] += 1;
            if parse_numeric(v).is_some() {
                numeric_ok[i] += 1;
            }
        }
    }

    let mut candidates = headers
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            if group_set.contains(name) {
                return None;
            }
            if name.eq_ignore_ascii_case("date")
                || name.ends_with("_id")
                || name.ends_with("_uuid")
                || name.ends_with("_url")
            {
                return None;
            }
            let ne = non_empty[i];
            if ne == 0 {
                return None;
            }
            let ratio = numeric_ok[i] as f64 / ne as f64;
            if ratio >= 0.8 {
                Some((name.to_string(), i, ne))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.2.cmp(&a.2));

    for (name, idx, _) in candidates {
        if out.iter().any(|(n, _)| n == &name) {
            continue;
        }
        out.push((name, idx));
        if out.len() >= max_metrics {
            break;
        }
    }

    if out.is_empty() && seen > 0 {
        return Err(anyhow!(
            "no numeric metric columns auto-detected; pass --metrics explicitly"
        ));
    }
    Ok(out)
}

fn parse_numeric(v: &str) -> Option<f64> {
    let s = v.replace(',', "").replace('$', "");
    s.parse::<f64>().ok()
}

fn parse_date_like(v: &str) -> Option<chrono::NaiveDate> {
    let s = v.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Some(d);
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.date());
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f") {
        return Some(dt.date());
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.date_naive());
    }
    None
}

fn markdown_to_html(markdown: &str) -> String {
    let mut out = String::new();
    let parser = MdParser::new_ext(markdown, Options::all());
    html::push_html(&mut out, parser);
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>FactorLens Report</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif;max-width:1024px;margin:24px auto;padding:0 16px;line-height:1.5}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left}}th{{background:#f5f5f5}}code{{background:#f3f3f3;padding:2px 4px;border-radius:4px}}</style></head><body>{}</body></html>",
        out
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn test_data_path(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data")
            .join(name)
    }

    fn read_csv_rows(path: &PathBuf) -> (StringRecord, Vec<StringRecord>) {
        let mut rdr = csv::Reader::from_path(path).expect("read csv");
        let headers = rdr.headers().expect("headers").clone();
        let rows = rdr
            .records()
            .map(|r| r.expect("row"))
            .collect::<Vec<_>>();
        (headers, rows)
    }

    #[test]
    fn dedup_driver_specs_keeps_first_label_instance() {
        let specs = vec![
            DriverSpec {
                label: "sum(a)".to_string(),
                col_idx: Some(1),
                agg: DriverAgg::Sum,
            },
            DriverSpec {
                label: "sum(a)".to_string(),
                col_idx: Some(2),
                agg: DriverAgg::Mean,
            },
            DriverSpec {
                label: "avg(b)".to_string(),
                col_idx: Some(3),
                agg: DriverAgg::Mean,
            },
        ];

        let out = dedup_driver_specs(specs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].label, "sum(a)");
        assert_eq!(out[0].col_idx, Some(1));
        assert!(matches!(out[0].agg, DriverAgg::Sum));
        assert_eq!(out[1].label, "avg(b)");
    }

    #[test]
    fn infer_numeric_driver_agg_detects_mean_like_columns() {
        assert!(matches!(
            infer_numeric_driver_agg("conversion_rate"),
            DriverAgg::Mean
        ));
        assert!(matches!(
            infer_numeric_driver_agg("discount_pct"),
            DriverAgg::Mean
        ));
        assert!(matches!(
            infer_numeric_driver_agg("avg_price_usd"),
            DriverAgg::Mean
        ));
        assert!(matches!(
            infer_numeric_driver_agg("traffic"),
            DriverAgg::Sum
        ));
    }

    #[test]
    fn auto_select_numeric_drivers_skips_count_like_columns() {
        let headers = StringRecord::from(vec![
            "quote_group_created_at",
            "customer_purchase_order_retail_total_price_usd",
            "net_gmv",
            "count_proposal_per_quote_group",
            "advantage_plan",
        ]);
        let rows = (0..40)
            .map(|i| {
                StringRecord::from(vec![
                    format!("2026-02-{:02}", (i % 28) + 1),
                    format!("{}", 1000.0 + (i as f64 * 10.0)),
                    format!("{}", 700.0 + (i as f64 * 8.0)),
                    format!("{}", 3 + (i % 5)),
                    format!("{}", i % 2),
                ])
            })
            .collect::<Vec<_>>();

        let selected = auto_select_numeric_drivers(&headers, &rows, 1, 0, 5, None);
        let names = selected.into_iter().map(|(name, _)| name).collect::<Vec<_>>();

        assert!(names.contains(&"net_gmv".to_string()));
        assert!(!names.contains(&"count_proposal_per_quote_group".to_string()));
        assert!(!names.contains(&"advantage_plan".to_string()));
    }

    #[test]
    fn investigate_both_paths_respects_json_out() {
        let out = PathBuf::from("artifacts/custom.json");
        let (md, json) = investigate_both_paths(&out);
        assert_eq!(md, PathBuf::from("artifacts/custom.md"));
        assert_eq!(json, PathBuf::from("artifacts/custom.json"));
    }

    #[test]
    fn investigate_both_paths_generates_json_for_md_out() {
        let out = PathBuf::from("artifacts/custom.md");
        let (md, json) = investigate_both_paths(&out);
        assert_eq!(md, PathBuf::from("artifacts/custom.md"));
        assert_eq!(json, PathBuf::from("artifacts/custom.json"));
    }

    #[test]
    fn default_output_prefixes_are_command_specific() {
        let analyze_args = AnalyzeArgs {
            input: Some(PathBuf::from("data/demo.csv")),
            postgres_url: None,
            query: None,
            query_file: None,
            postgres_ssl_mode: PostgresSslMode::Prefer,
            postgres_ca_file: None,
            profile: None,
            profile_config: None,
            group_by: vec![],
            auto_group_k: 3,
            metrics: vec![],
            count_only: false,
            agg: AggKind::Sum,
            percentiles: vec![],
            normalize_text_groups: false,
            word_freq: false,
            date_column: None,
            time_grain: None,
            period: None,
            anchor_date: None,
            current_start: None,
            current_end: None,
            previous_start: None,
            previous_end: None,
            r#where: vec![],
            exclude_blank_groups: false,
            rank_by: None,
            top: 20,
            top_insights: 0,
            opportunity_min_records: 2,
            min_records: 1,
            alert_top5_share: None,
            alert_blank_share: None,
            alert_rule: vec![],
            output_format: OutputFormat::Both,
            out: None,
        };
        assert_eq!(
            default_analyze_out(&analyze_args),
            PathBuf::from("artifacts/analyze_demo.md")
        );

        let investigate_args = InvestigateArgs {
            input: PathBuf::from("data/demo.csv"),
            metric: "revenue".to_string(),
            drivers: vec![],
            driver_preset: None,
            auto_drivers: AutoDriversMode::Deterministic,
            dedup_drivers: true,
            driver_contrib: InvestigateContribMode::Percent,
            top_drivers: 3,
            output_format: InvestigateOutputFormat::Both,
            out: None,
            max_id_drivers: 3,
            max_cat_drivers: 2,
            max_num_drivers: 2,
            date_column: None,
            time_grain: None,
            period: None,
            anchor_date: None,
            current_start: None,
            current_end: None,
            previous_start: None,
            previous_end: None,
        };
        assert_eq!(
            default_analyze_investigate_out(&investigate_args),
            PathBuf::from("artifacts/investigate_demo.md")
        );
    }

    #[test]
    fn analyze_drivers_identity_prediction_matches_demo_metric_total() {
        let path = test_data_path("demo_revenue.csv");
        let (headers, rows) = read_csv_rows(&path);
        let metric_idx = headers.iter().position(|h| h == "revenue_usd").unwrap();
        let identity = infer_driver_identity(&headers, &rows, &rows, metric_idx)
            .expect("infer ok")
            .expect("identity found");

        let actual = aggregate_column(&rows, metric_idx, DriverAgg::Sum);
        let predicted = aggregate_identity_prediction(&rows, &identity);
        assert!((actual - predicted).abs() < 1e-6, "actual={} predicted={}", actual, predicted);
    }

    #[test]
    fn residual_summary_is_suppressed_for_clean_demo() {
        let path = test_data_path("demo_revenue.csv");
        let (headers, rows) = read_csv_rows(&path);
        let metric_idx = headers.iter().position(|h| h == "revenue_usd").unwrap();
        let identity = infer_driver_identity(&headers, &rows, &rows, metric_idx)
            .expect("infer ok")
            .expect("identity found");

        let summary = analyze_drivers_residual_summary(
            &headers,
            &rows,
            &rows,
            metric_idx,
            &identity,
            0.1,
            5_970.47,
        );
        assert!(summary.signals.is_empty());
    }

    #[test]
    fn residual_summary_surfaces_hidden_segment_in_residual_demo() {
        let path = test_data_path("demo_revenue_residual.csv");
        let (headers, rows) = read_csv_rows(&path);
        let metric_idx = headers.iter().position(|h| h == "revenue_usd").unwrap();
        let identity = infer_driver_identity(&headers, &rows, &rows, metric_idx)
            .expect("infer ok")
            .expect("identity found");

        let summary = analyze_drivers_residual_summary(
            &headers,
            &rows,
            &rows,
            metric_idx,
            &identity,
            1.5,
            77_765.73,
        );
        assert!(!summary.signals.is_empty());
        assert!(
            summary
                .signals
                .iter()
                .any(|s| s.name.contains("campaign = spring_launch")),
            "signals={:?}",
            summary.signals
        );
    }

    #[test]
    fn investigate_residual_model_uses_heuristic_mode_without_numeric_drivers() {
        let path = test_data_path("demo_revenue_residual.csv");
        let (headers, rows) = read_csv_rows(&path);
        let metric_idx = headers.iter().position(|h| h == "revenue_usd").unwrap();
        let date_idx = headers.iter().position(|h| h == "date").unwrap();
        let period_cfg = PeriodCompareConfig {
            date_column: "date".to_string(),
            time_grain: Some(TimeGrain::Month),
            period: Some(PeriodPreset::Last),
            current_start: chrono::NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
            current_end: chrono::NaiveDate::from_ymd_opt(2026, 3, 31).unwrap(),
            previous_start: chrono::NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
            previous_end: chrono::NaiveDate::from_ymd_opt(2026, 2, 28).unwrap(),
        };
        let driver_specs = vec![DriverSpec {
            label: "count_distinct(channel)".to_string(),
            col_idx: headers.iter().position(|h| h == "channel"),
            agg: DriverAgg::CountDistinct,
        }];

        let model = investigate_residual_summary(
            &headers,
            &rows,
            metric_idx,
            date_idx,
            &period_cfg,
            &driver_specs,
            1.0,
            1.0,
        );
        assert_eq!(model.decomposition_mode, "heuristic");
        assert!(model.driver_contributions.is_empty());
    }
}

fn build_analysis_prompt_context(v: &serde_json::Value, evidence: &[String]) -> String {
    let records = v.get("records").and_then(|x| x.as_u64()).unwrap_or(0);
    let segments = v.get("segments").and_then(|x| x.as_u64()).unwrap_or(0);
    let group_by = v
        .get("group_by")
        .and_then(|x| x.as_array())
        .map(|xs| {
            xs.iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_else(|| "unknown".to_string());
    let metrics = v
        .get("metrics")
        .and_then(|x| x.as_array())
        .map(|xs| {
            xs.iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_else(|| "unknown".to_string());
    let alerts = v
        .get("alerts")
        .and_then(|x| x.as_array())
        .map(|xs| {
            xs.iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        })
        .unwrap_or_else(|| "".to_string());
    let evidence_block = if evidence.is_empty() {
        "none".to_string()
    } else {
        evidence
            .iter()
            .enumerate()
            .map(|(i, s)| format!("[E{}] {}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "records={} | segments={} | group_by={} | metrics={}\nalerts={}\nevidence:\n{}",
        records,
        segments,
        group_by,
        metrics,
        if alerts.is_empty() { "none" } else { &alerts },
        evidence_block
    )
}

fn build_analysis_evidence(v: &serde_json::Value) -> Vec<String> {
    let mut out = Vec::new();
    let top_groups = v
        .get("groups")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let primary_metric = v
        .get("primary_metric")
        .and_then(|x| x.as_str())
        .unwrap_or("primary_metric");

    for g in top_groups.iter().take(8) {
        let name = g
            .get("group")
            .and_then(|x| x.as_str())
            .unwrap_or("(unknown)");
        let count = g.get("count").and_then(|x| x.as_u64()).unwrap_or(0);
        let count_share = g
            .get("count_share_pct")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        let primary_val = g
            .get(primary_metric)
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        out.push(format!(
            "group='{}' records={} record_share_pct={:.1} {}={}",
            name,
            count,
            count_share,
            primary_metric,
            fmt_num(primary_val, 2)
        ));
    }

    let alerts = v
        .get("alerts")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    for a in alerts {
        if let Some(s) = a.as_str() {
            out.push(format!("alert='{}'", s));
        }
    }

    out
}

fn should_normalize_group_column(col: &str) -> bool {
    let c = col.to_lowercase();
    c.contains("name") || c.contains("title")
}

fn normalize_group_value(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut prev_space = false;
    for ch in lower.chars() {
        let keep = ch.is_alphanumeric() || ch.is_whitespace();
        if !keep {
            continue;
        }
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

fn tokenize_words(s: &str) -> Vec<String> {
    let normalized = normalize_group_value(s);
    const STOP: &[&str] = &[
        "the", "and", "for", "of", "to", "in", "on", "with", "a", "an", "or", "by", "from",
        "per", "will", "assumes", "assume", "data",
    ];
    normalized
        .split_whitespace()
        .filter_map(|w| {
            if w.len() < 3 || STOP.contains(&w) || w.chars().all(|c| c.is_ascii_digit()) {
                None
            } else {
                Some(w.to_string())
            }
        })
        .collect()
}

fn is_blank_group_key(group: &str) -> bool {
    group.split(" | ").all(|part| part.trim() == "(blank)")
}

fn is_effectively_blank(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return true;
    }
    t.chars().all(|c| !c.is_alphanumeric())
}

fn top_word_counts(counts: &HashMap<String, u64>, top_n: usize) -> Vec<(String, u64)> {
    let mut v = counts
        .iter()
        .map(|(w, c)| (w.clone(), *c))
        .collect::<Vec<_>>();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v.truncate(top_n);
    v
}

fn aggregate_values(values: &[f64], agg: AggKind) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    match agg {
        AggKind::Sum => values.iter().sum::<f64>(),
        AggKind::Mean => values.iter().sum::<f64>() / values.len() as f64,
        AggKind::Median => {
            let mut xs = values.to_vec();
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = xs.len();
            if n % 2 == 1 {
                xs[n / 2]
            } else {
                (xs[n / 2 - 1] + xs[n / 2]) / 2.0
            }
        }
    }
}

fn percentile_value(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    let mut xs = values.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = q.clamp(0.0, 1.0);
    let rank = q * (xs.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        xs[lo]
    } else {
        let w = rank - lo as f64;
        xs[lo] * (1.0 - w) + xs[hi] * w
    }
}

fn pseudo_rand_below(state: &mut u64, upper_exclusive: u64) -> u64 {
    // Deterministic LCG-based pseudo-random for lightweight sampling.
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1);
    if upper_exclusive == 0 {
        0
    } else {
        *state % upper_exclusive
    }
}

fn fmt_num(value: f64, decimals: usize) -> String {
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let s = format!("{:.*}", decimals, value.abs());
    let (int_part, frac_part) = s.split_once('.').unwrap_or((&s, ""));

    let mut grouped_rev = String::with_capacity(int_part.len() + int_part.len() / 3);
    for (i, ch) in int_part.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            grouped_rev.push('_');
        }
        grouped_rev.push(ch);
    }
    let grouped_int = grouped_rev.chars().rev().collect::<String>();

    if decimals == 0 {
        format!("{}{}", sign, grouped_int)
    } else {
        format!("{}{}.{}", sign, grouped_int, frac_part)
    }
}

fn fmt_num_commas(value: f64, decimals: usize) -> String {
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let s = format!("{:.*}", decimals, value.abs());
    let (int_part, frac_part) = s.split_once('.').unwrap_or((&s, ""));

    let mut grouped_rev = String::with_capacity(int_part.len() + int_part.len() / 3);
    for (i, ch) in int_part.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            grouped_rev.push(',');
        }
        grouped_rev.push(ch);
    }
    let grouped_int = grouped_rev.chars().rev().collect::<String>();

    if decimals == 0 {
        format!("{}{}", sign, grouped_int)
    } else {
        format!("{}{}.{}", sign, grouped_int, frac_part)
    }
}

fn signed_fmt_num_commas(value: f64, decimals: usize) -> String {
    let sign = if value >= 0.0 { "+" } else { "-" };
    format!("{}{}", sign, fmt_num_commas(value.abs(), decimals))
}

fn pretty_signal_name(name: &str) -> String {
    if name.contains(" = ") {
        return name.to_string();
    }
    if let Some((left, right)) = name.split_once('=') {
        format!("{} = {}", left, right)
    } else {
        name.to_string()
    }
}

#[derive(Debug)]
struct ParsedAlertRule {
    metric: String,
    op: String,
    threshold: f64,
}

#[derive(Debug)]
struct AlertEvalContext {
    top5_record_share_pct: f64,
    blank_share_pct: f64,
    segments: f64,
    records: f64,
}

fn parse_alert_rule(raw: &str) -> Result<ParsedAlertRule> {
    let s = raw.trim();
    if s.is_empty() {
        return Err(anyhow!("empty --alert-rule"));
    }
    let ops = [">=", "<=", "!=", ">", "<", "="];
    for op in ops {
        if let Some((lhs, rhs)) = s.split_once(op) {
            let metric = lhs.trim().to_lowercase();
            if metric.is_empty() {
                return Err(anyhow!("invalid --alert-rule '{}': missing metric", s));
            }
            if !is_supported_alert_metric(&metric) {
                return Err(anyhow!(
                    "invalid --alert-rule '{}': unsupported metric '{}'. Supported: top5_record_share_pct, blank_share_pct, segments, records",
                    s,
                    metric
                ));
            }
            let threshold = rhs
                .trim()
                .parse::<f64>()
                .map_err(|_| anyhow!("invalid --alert-rule '{}': bad threshold", s))?;
            return Ok(ParsedAlertRule {
                metric,
                op: op.to_string(),
                threshold,
            });
        }
    }
    Err(anyhow!(
        "invalid --alert-rule '{}'; expected metric<op>number (e.g., top5_record_share_pct>60)",
        s
    ))
}

fn is_supported_alert_metric(metric: &str) -> bool {
    matches!(
        metric,
        "top5_record_share_pct"
            | "top5_share"
            | "top5_record_share"
            | "blank_share_pct"
            | "blank_share"
            | "segments"
            | "records"
    )
}

fn alert_metric_value(metric: &str, ctx: &AlertEvalContext) -> f64 {
    match metric {
        "top5_record_share_pct" | "top5_share" | "top5_record_share" => ctx.top5_record_share_pct,
        "blank_share_pct" | "blank_share" => ctx.blank_share_pct,
        "segments" => ctx.segments,
        "records" => ctx.records,
        _ => f64::NAN,
    }
}

fn eval_alert_rule(rule: &ParsedAlertRule, ctx: &AlertEvalContext) -> bool {
    let actual = alert_metric_value(&rule.metric, ctx);
    if actual.is_nan() {
        return false;
    }
    match rule.op.as_str() {
        ">" => actual > rule.threshold,
        ">=" => actual >= rule.threshold,
        "<" => actual < rule.threshold,
        "<=" => actual <= rule.threshold,
        "=" => (actual - rule.threshold).abs() < 1e-9,
        "!=" => (actual - rule.threshold).abs() >= 1e-9,
        _ => false,
    }
}

fn parse_where_filters(
    clauses: &[String],
    headers: &StringRecord,
) -> Result<Vec<(String, usize, String)>> {
    let mut out = Vec::new();
    for clause in clauses {
        let raw = clause.trim();
        if raw.is_empty() {
            continue;
        }
        let (col_raw, value_raw) = raw
            .split_once('=')
            .ok_or_else(|| anyhow!("invalid --where clause '{}'; expected column=value", raw))?;
        let col = resolve_group_name(col_raw.trim(), headers)?;
        let idx = headers
            .iter()
            .position(|h| h == col)
            .ok_or_else(|| anyhow!("filter column not found: {}", col))?;
        out.push((col, idx, value_raw.trim().to_string()));
    }
    Ok(out)
}

fn matches_where_filters(rec: &StringRecord, filters: &[(String, usize, String)]) -> bool {
    filters.iter().all(|(_, idx, want)| {
        let got = rec.get(*idx).unwrap_or("").trim();
        got == want
    })
}
