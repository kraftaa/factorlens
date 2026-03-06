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
use rustls::ClientConfig;
use serde::Deserialize;
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
    Analyze(AnalyzeArgs),
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
    #[arg(long, value_enum, default_value = "sum")]
    agg: AggKind,
    #[arg(long, value_delimiter = ',')]
    percentiles: Vec<PercentileKind>,
    #[arg(long, default_value_t = false)]
    normalize_text_groups: bool,
    #[arg(long, default_value_t = false)]
    word_freq: bool,
    #[arg(long, value_delimiter = ',')]
    r#where: Vec<String>,
    #[arg(long)]
    rank_by: Option<String>,
    #[arg(long, default_value_t = 20)]
    top: usize,
    #[arg(long, default_value_t = 1)]
    min_records: u64,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_enum, default_value = "both")]
    output_format: OutputFormat,
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
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum PostgresSslMode {
    Disable,
    Prefer,
    Require,
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
    let (input_path, _temp_path) = materialize_analyze_input(&args)?;
    let report = analyze_table_csv(
        &input_path,
        args.profile.as_deref(),
        &args.group_by,
        args.auto_group_k,
        &args.metrics,
        args.agg,
        &args.percentiles,
        args.normalize_text_groups,
        args.word_freq,
        &args.r#where,
        args.rank_by.as_deref(),
        args.top,
        args.min_records,
    )?;
    match args.output_format {
        OutputFormat::Md => {
            fs::write(&args.out, report.markdown)?;
            println!("Analysis (markdown) written to {}", args.out.display());
        }
        OutputFormat::Json => {
            fs::write(&args.out, serde_json::to_string_pretty(&report.json)?)?;
            println!("Analysis (json) written to {}", args.out.display());
        }
        OutputFormat::Both => {
            fs::write(&args.out, report.markdown)?;
            let json_path = args.out.with_extension("json");
            fs::write(&json_path, serde_json::to_string_pretty(&report.json)?)?;
            println!("Analysis written to {}", args.out.display());
            println!("Analysis JSON written to {}", json_path.display());
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
    agg: AggKind,
    percentiles: &[PercentileKind],
    normalize_text_groups: bool,
    word_freq: bool,
    where_clauses: &[String],
    rank_by: Option<&str>,
    top_n: usize,
    min_records: u64,
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

    let metric_cols = if metrics.is_empty() {
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
    let rank_metric = rank_by.and_then(|rb| {
        metric_cols
            .iter()
            .find(|(m, _)| m == rb)
            .map(|(m, _)| m.clone())
    });
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

    let group_idxs = resolved_groups
        .iter()
        .map(|g| {
            headers
                .iter()
                .position(|h| h == g)
                .ok_or_else(|| anyhow!("group column not found: {}", g))
        })
        .collect::<Result<Vec<_>>>()?;
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

    let mut by_group: BTreeMap<String, (u64, HashMap<String, Vec<f64>>)> = BTreeMap::new();
    let mut word_counts: HashMap<String, u64> = HashMap::new();
    let mut row_count = 0_u64;

    for rec in rdr.records() {
        let rec = rec?;
        if !matches_where_filters(&rec, &where_filters) {
            continue;
        }
        row_count += 1;
        let gk = group_idxs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                let raw = rec.get(*idx).unwrap_or("").trim();
                let mut v = if normalize_text_groups
                    && should_normalize_group_column(&resolved_groups[i])
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
            .join(" | ");

        if word_freq {
            for (i, _) in &word_group_cols {
                let raw = rec.get(group_idxs[*i]).unwrap_or("").trim();
                for w in tokenize_words(raw) {
                    *word_counts.entry(w).or_insert(0) += 1;
                }
            }
        }

        let entry = by_group
            .entry(gk)
            .or_insert_with(|| (0, HashMap::<String, Vec<f64>>::new()));
        entry.0 += 1;

        for (name, idx) in &metric_cols {
            let raw = rec.get(*idx).unwrap_or("").trim();
            if let Some(v) = parse_numeric(raw) {
                entry.1.entry(name.clone()).or_default().push(v);
            }
        }

    }

    let primary_metric = metric_cols
        .first()
        .map(|(m, _)| m.clone())
        .or_else(|| metric_cols.first().map(|(m, _)| m.clone()));
    let total_count_all = by_group.values().map(|(c, _)| *c).sum::<u64>();
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
        "- Ranking: {}\n",
        rank_metric.clone().unwrap_or_else(|| "count".to_string())
    ));
    md.push_str(&format!("- Top rows shown: {}\n", top_n));
    md.push_str(&format!("- Minimum records per segment: {}\n", min_records));
    md.push_str(&format!("- Metric aggregation: {}\n", agg.label()));
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
        md.push_str("- Metrics: none detected (count-only analysis)\n\n");
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
    md.push('\n');

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
        "metrics": display_metrics,
        "base_metrics": metric_cols.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        "metric_aggregation": agg.label().to_lowercase(),
        "percentiles": percentiles.iter().map(|p| p.label()).collect::<Vec<_>>(),
        "normalize_text_groups": normalize_text_groups,
        "word_frequency": top_words
            .iter()
            .map(|(w, c)| serde_json::json!({"word": w, "count": c}))
            .collect::<Vec<_>>(),
        "rank_by": rank_metric.clone().unwrap_or_else(|| "count".to_string()),
        "top": top_n,
        "min_records": min_records,
        "primary_metric": primary_metric,
        "top5_count": top5_count,
        "top5_primary_metric_value": top5_primary,
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
