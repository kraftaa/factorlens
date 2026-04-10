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
use llm_local::{build_client, Backend, LlmClient};
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
use std::path::{Path, PathBuf};
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
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        strict_facts: bool,
        #[arg(long, default_value_t = 5)]
        max_bullets: usize,
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
    /// Legacy/specialized numeric-driver decomposition (prefer `investigate` for new workflows).
    AnalyzeInvestigate(InvestigateArgs),
    AnalyzeDrivers(AnalyzeDriversArgs),
    AnalyzeSuggest(AnalyzeSuggestArgs),
    AnalyzeCompare(AnalyzeCompareArgs),
    /// Guided multi-step change investigation across snapshots/periods.
    Investigate(InvestigateWorkflowArgs),
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
struct InvestigateWorkflowArgs {
    #[arg(long, default_value_t = String::new())]
    question: String,
    #[arg(long, value_enum)]
    mode: Option<InvestigationModeArg>,
    #[arg(long, conflicts_with_all = ["profile", "profile_config"])]
    config: Option<PathBuf>,
    #[arg(long)]
    profile: Option<String>,
    #[arg(long, requires = "profile", conflicts_with = "config")]
    profile_config: Option<PathBuf>,
    #[arg(long)]
    base: Option<PathBuf>,
    #[arg(long)]
    new: Option<PathBuf>,
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
    metric: Option<String>,
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
    dimensions: Vec<String>,
    #[arg(long, value_delimiter = ',')]
    drill_fields: Vec<String>,
    #[arg(long, default_value_t = 2)]
    max_depth: usize,
    #[arg(long, default_value_t = 3)]
    max_branches: usize,
    #[arg(long, default_value_t = 5.0)]
    min_contribution: f64,
    #[arg(long, default_value_t = 0.0)]
    min_delta_abs: f64,
    #[arg(long, default_value_t = 0.0)]
    min_score_improvement: f64,
    #[arg(long, default_value_t = 5)]
    min_slice_rows: u64,
    #[arg(long, default_value_t = 12)]
    top_movers: usize,
    #[arg(long, value_enum, default_value = "deterministic")]
    planner: InvestigationPlanner,
    #[arg(long, value_enum, default_value = "local")]
    planner_backend: BackendArg,
    #[arg(long)]
    planner_model: Option<String>,
    #[arg(long, default_value_t = false)]
    verbose: bool,
    #[arg(long, default_value_t = false)]
    trace: bool,
    #[arg(long, value_enum, default_value = "both")]
    output_format: InvestigateOutputFormat,
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
    #[arg(long, value_enum, default_value = "toml")]
    profile_format: SuggestProfileFormat,
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
    #[arg(long, value_delimiter = ',')]
    driver_include: Vec<String>,
    #[arg(long, value_delimiter = ',')]
    driver_exclude: Vec<String>,
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
enum SuggestProfileFormat {
    Toml,
    Json,
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize)]
enum InvestigationMode {
    ChangeDrivers,
    ConcentrationDrivers,
    CompareSnapshots,
    RecommendNext,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum InvestigationPlanner {
    Deterministic,
    Llm,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum InvestigationModeArg {
    #[value(name = "change_drivers", alias = "change-drivers")]
    ChangeDrivers,
    #[value(name = "concentration_drivers", alias = "concentration-drivers")]
    ConcentrationDrivers,
    #[value(name = "compare_snapshots", alias = "compare-snapshots")]
    CompareSnapshots,
    #[value(name = "recommend_next", alias = "recommend-next")]
    RecommendNext,
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
            strict_facts,
            max_bullets,
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
            let system = "You are an analytics assistant. Use only provided analysis context. If missing, say unknown. Respond in plain text bullets only. Cite evidence IDs like [E1], [E2] for each factual claim. Use at most 2 decimal places.";
            let user = format!("Question: {}\n\nAnalysis context:\n{}", question, context);
            let raw = client.answer(system, &user)?;
            let max_bullets = max_bullets.max(1);
            let mut answer = sanitize_explain_analyze_answer(&raw, max_bullets);
            if strict_facts && !explain_analyze_answer_is_usable(&answer, &evidence) {
                answer =
                    deterministic_explain_analyze_from_evidence(&question, &evidence, max_bullets);
            }
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
        Commands::Investigate(args) => {
            run_investigate_workflow(args)?;
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
        return Err(anyhow!("use either --drivers or --driver-preset, not both"));
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
    let date_idx = date_col_name
        .as_ref()
        .and_then(|n| headers.iter().position(|h| h == n));
    let didx = date_idx.ok_or_else(|| {
        anyhow!(
            "date column is required for investigate mode; pass --date-column or provide a detectable date column"
        )
    })?;

    let period_cfg =
        parse_period_cfg_from_investigate(&args, date_col_name.clone().unwrap_or_default())?;

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
    driver_specs = apply_driver_filters(driver_specs, &args.driver_include, &args.driver_exclude);
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
                let d =
                    *driver_curr.get(name).unwrap_or(&0.0) - *driver_prev.get(name).unwrap_or(&0.0);
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
    println!(
        "- explained: {:+.1}% ({:.0}%)",
        explained_pct, explained_share_pct
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
            args.input_new
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        )
    } else {
        let cfg = parse_period_cfg_from_driver_args(&args, headers.clone())?;
        format!(
            "{}..{} vs {}..{}",
            cfg.current_start, cfg.current_end, cfg.previous_start, cfg.previous_end
        )
    };

    let identity_text = match identity.op {
        DriverIdentityOp::Multiply => format!(
            "{} ≈ {} * {}",
            identity.metric, identity.left, identity.right
        ),
        DriverIdentityOp::Divide => format!(
            "{} ≈ {} / {}",
            identity.metric, identity.left, identity.right
        ),
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
    println!(
        "- {}: {:+.1}%",
        driver_identity_name(&identity.left),
        left_contrib
    );
    println!(
        "- {}: {:+.1}%",
        driver_identity_name(&identity.right),
        right_contrib
    );
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
            println!(
                "Analyze drivers (markdown) written to {}",
                out_path.display()
            );
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
        return Err(anyhow!(
            "--date-column is required when --input-new is not provided"
        ));
    };
    let temp = InvestigateArgs {
        input: args.input.clone(),
        metric: args.metric.clone(),
        drivers: vec![],
        driver_include: vec![],
        driver_exclude: vec![],
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
                let (mape, rows_used) = candidate_identity_fit(
                    all_rows.iter().copied(),
                    metric_idx,
                    *left_idx,
                    *right_idx,
                    op,
                );
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

#[allow(clippy::too_many_arguments)]
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
                        detail: format!(
                            "{} values correlate with residual (corr={:+.2})",
                            direction, corr
                        ),
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
                .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
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

#[allow(clippy::too_many_arguments)]
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
        DriverIdentityOp::Multiply => format!(
            "{} ≈ {} * {}",
            identity.metric, identity.left, identity.right
        ),
        DriverIdentityOp::Divide => format!(
            "{} ≈ {} / {}",
            identity.metric, identity.left, identity.right
        ),
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
    md.push_str(&format!(
        "## {} change: {:+.1}%\n\n",
        identity.metric, total_change_pct
    ));
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
            md.push_str(&format!(
                "- {}: {}\n",
                pretty_signal_name(&signal.name),
                signal.detail
            ));
        }
    }
    md.push_str("\n## Artifacts written\n\n");
    let (md_path, json_path) = investigate_both_paths(
        &args
            .out
            .clone()
            .unwrap_or_else(|| default_analyze_drivers_out(args)),
    );
    md.push_str(&format!("- {}\n", md_path.display()));
    md.push_str(&format!("- {}\n", json_path.display()));
    md
}

#[allow(clippy::too_many_arguments)]
fn render_investigate_markdown(
    input: &Path,
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
    md.push_str(&format!(
        "## {} change: {:+.2}%\n\n",
        metric, metric_change_pct
    ));
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
    md.push_str(&format!(
        "- explained: {:+.2}% ({:.0}%)\n",
        explained_pct, explained_share_pct
    ));
    md.push_str(&format!(
        "- residual: {:+.2}% ({}).\n",
        residual.residual_pct,
        signed_fmt_num_commas(residual.residual_amount, 2)
    ));
    if !residual.signals.is_empty() {
        md.push_str("\n## Residual segments\n\n");
        for signal in &residual.signals {
            md.push_str(&format!(
                "- {}: {}\n",
                pretty_signal_name(&signal.name),
                signal.detail
            ));
        }
    }
    md
}

fn investigate_both_paths(out_path: &Path) -> (PathBuf, PathBuf) {
    let ext = out_path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "json" {
        (out_path.with_extension("md"), out_path.to_path_buf())
    } else {
        (out_path.to_path_buf(), out_path.with_extension("json"))
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

fn apply_driver_filters(
    mut specs: Vec<DriverSpec>,
    driver_include: &[String],
    driver_exclude: &[String],
) -> Vec<DriverSpec> {
    let matches_filter = |label: &str, set: &HashSet<String>| -> bool {
        let normalized = label.to_ascii_lowercase();
        if set.contains(&normalized) {
            return true;
        }
        if let Some((_, arg)) = parse_driver_fn(label) {
            return set.contains(&arg.to_ascii_lowercase());
        }
        false
    };

    if !driver_include.is_empty() {
        let include_set = driver_include
            .iter()
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect::<HashSet<_>>();
        if !include_set.is_empty() {
            specs.retain(|d| matches_filter(&d.label, &include_set));
        }
    }
    if !driver_exclude.is_empty() {
        let exclude_set = driver_exclude
            .iter()
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect::<HashSet<_>>();
        specs.retain(|d| !matches_filter(&d.label, &exclude_set));
    }
    specs
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
            if (lname == "id" || lname.ends_with("_id") || lname.ends_with("_uuid")) && !is_uuid_dup
            {
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
            &[
                "name",
                "title",
                "description",
                "comment",
                "note",
                "email",
                "address",
            ],
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

fn is_text_identity_like_column(col_name: &str) -> bool {
    // Columns that look like free-text identifiers or contact info should not be treated as numeric drivers.
    let n = col_name.to_ascii_lowercase();
    n.contains("email")
        || n.contains("name")
        || n.contains("address")
        || n.contains("phone")
        || n.contains("url")
        || n.contains("http")
        || n.contains("uuid")
        || n.contains("guid")
        || n.contains("ip")
        || n.contains("contact")
}

fn auto_select_numeric_drivers(
    headers: &StringRecord,
    rows: &[StringRecord],
    metric_idx: usize,
    date_idx: usize,
    top_n: usize,
    dedup_corr_threshold: Option<f64>,
) -> Vec<(String, usize)> {
    if top_n == 0 {
        return vec![];
    }
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
            || is_text_identity_like_column(&lname)
        {
            continue;
        }
        let (distinct_count, non_empty) = approx_distinct_count(rows, idx, 5000);
        if non_empty < 20 {
            continue;
        }
        if distinct_count <= 3 || (is_indicator_like_numeric_column(&lname) && distinct_count <= 10)
        {
            continue;
        }
        let mut xs = Vec::<f64>::new();
        let mut ys = Vec::<f64>::new();
        let mut parsed_driver = 0usize;
        let mut seen = 0usize;
        for rec in rows.iter().take(5000) {
            let raw_x = rec.get(idx).unwrap_or("").trim();
            let raw_y = rec.get(metric_idx).unwrap_or("").trim();
            let x = parse_numeric(raw_x);
            if !raw_x.is_empty() {
                seen += 1;
                if x.is_some() {
                    parsed_driver += 1;
                }
            }
            let y = parse_numeric(raw_y);
            if let (Some(x), Some(y)) = (x, y) {
                xs.push(x);
                ys.push(y);
            }
        }
        // Require at least 60% of non-empty values to be parseable as numeric to avoid misclassifying text columns.
        if seen > 0 && (parsed_driver as f64) / (seen as f64) < 0.6 {
            continue;
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
    let mut dedup_samples = HashMap::<usize, Vec<Option<f64>>>::new();
    for (name, idx, _corr_to_metric) in scored {
        let mut candidate_sample: Option<Vec<Option<f64>>> = None;
        let is_duplicate = if let Some(thr) = dedup_corr_threshold {
            candidate_sample = Some(numeric_column_sample(rows, idx));
            let sample = candidate_sample.as_ref().expect("sample is set");
            selected.iter().any(|(_, prev_idx)| {
                let prev_sample = dedup_samples
                    .entry(*prev_idx)
                    .or_insert_with(|| numeric_column_sample(rows, *prev_idx));
                let c = corr_between_numeric_samples(sample, prev_sample).abs();
                c.is_finite() && c >= thr
            })
        } else {
            false
        };
        if is_duplicate {
            continue;
        }
        if let Some(sample) = candidate_sample {
            dedup_samples.insert(idx, sample);
        }
        selected.push((name, idx));
        if selected.len() >= top_n {
            break;
        }
    }
    selected
}

fn numeric_column_sample(rows: &[StringRecord], col_idx: usize) -> Vec<Option<f64>> {
    rows.iter()
        .take(5000)
        .map(|rec| parse_numeric(rec.get(col_idx).unwrap_or("").trim()))
        .collect::<Vec<_>>()
}

fn corr_between_numeric_samples(a: &[Option<f64>], b: &[Option<f64>]) -> f64 {
    let mut n: usize = 0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut sum_xy = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        if let (Some(x), Some(y)) = (x, y) {
            n += 1;
            sum_x += *x;
            sum_y += *y;
            sum_x2 += *x * *x;
            sum_y2 += *y * *y;
            sum_xy += *x * *y;
        }
    }
    if n < 20 {
        0.0
    } else {
        let n_f = n as f64;
        let num = (n_f * sum_xy) - (sum_x * sum_y);
        let den_x = (n_f * sum_x2) - (sum_x * sum_x);
        let den_y = (n_f * sum_y2) - (sum_y * sum_y);
        let den = (den_x * den_y).sqrt();
        if den <= 1e-12 {
            0.0
        } else {
            num / den
        }
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
    if den <= 1e-12 {
        0.0
    } else {
        num / den
    }
}

fn parse_period_cfg_from_investigate(
    args: &InvestigateArgs,
    date_column: String,
) -> Result<PeriodCompareConfig> {
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
            let cs = parse_date_arg(args.current_start.as_deref(), "--current-start is required")?;
            let ce = parse_date_arg(args.current_end.as_deref(), "--current-end is required")?;
            let ps = parse_date_arg(
                args.previous_start.as_deref(),
                "--previous-start is required",
            )?;
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

fn ensure_parent_dir(path: &Path) -> Result<()> {
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
    let profile_body = build_suggested_profile_config(
        &report,
        &args.profile_name,
        args.auto_group_k,
        args.max_metrics,
        args.profile_format,
    );
    let profile_path = args
        .out_profile
        .clone()
        .unwrap_or_else(|| default_suggest_profile_path(&args.out, args.profile_format));
    fs::write(&profile_path, profile_body)?;

    match args.output_format {
        SuggestOutputFormat::Md => {
            fs::write(&args.out, suggest_report_markdown(&report, &profile_path))?;
            println!(
                "Analyze suggest report (markdown) written to {}",
                args.out.display()
            );
        }
        SuggestOutputFormat::Json => {
            fs::write(&args.out, serde_json::to_string_pretty(&report)?)?;
            println!(
                "Analyze suggest report (json) written to {}",
                args.out.display()
            );
        }
        SuggestOutputFormat::Both => {
            fs::write(&args.out, suggest_report_markdown(&report, &profile_path))?;
            let json_path = args.out.with_extension("json");
            fs::write(&json_path, serde_json::to_string_pretty(&report)?)?;
            println!(
                "Analyze suggest report (markdown) written to {}",
                args.out.display()
            );
            println!(
                "Analyze suggest report (json) written to {}",
                json_path.display()
            );
        }
    }
    println!(
        "Suggested profile {} written to {}",
        suggest_profile_format_label(args.profile_format),
        profile_path.display()
    );
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
        let inferred_role =
            infer_column_role(&name, fill_pct, distinct_count, numeric_ratio, date_ratio);
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
        a_penalty.cmp(&b_penalty).then_with(|| {
            b.fill_pct
                .partial_cmp(&a.fill_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
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
        .max_by(|a, b| {
            a.fill_pct
                .partial_cmp(&b.fill_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|c| c.name.clone());

    let mut warnings = Vec::new();
    if suggested_group_by.is_empty() {
        warnings
            .push("No strong dimension columns detected. Pass --group-by manually.".to_string());
    }
    if suggested_metrics.is_empty() {
        warnings.push("No strong metric columns detected. Pass --metrics manually.".to_string());
    }
    for c in columns
        .iter()
        .filter(|c| looks_like_identifier_column(&c.name) && c.fill_pct >= 20.0)
    {
        warnings.push(format!(
            "Identifier-like column '{}' was excluded from suggestions.",
            c.name
        ));
    }
    for c in columns
        .iter()
        .filter(|c| c.fill_pct < 30.0 && c.inferred_role == "dimension")
    {
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
    let id_like = looks_like_identifier_column(&n)
        || n.contains("uuid")
        || n.contains("guid")
        || n.ends_with("_url")
        || n.ends_with("_uri");
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
    if n.contains("qty") || n.contains("quantity") || n.contains("count") || n.contains("orders") {
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

fn build_suggested_profile_json(
    report: &AnalyzeSuggestReport,
    profile_name: &str,
    auto_group_k: usize,
    max_metrics: usize,
) -> String {
    let group_by = report.suggested_group_by.clone();
    let metrics = report
        .suggested_metrics
        .iter()
        .take(max_metrics)
        .cloned()
        .collect::<Vec<_>>();
    let rank_by = report.suggested_rank_by.clone();
    serde_json::json!({
        "profiles": {
            profile_name: {
                "group_by": group_by,
                "metrics": metrics,
                "rank_by": rank_by,
                "top": 15,
                "min_records": 10,
                "auto_group_k": auto_group_k,
            }
        }
    })
    .to_string()
}

fn build_suggested_profile_config(
    report: &AnalyzeSuggestReport,
    profile_name: &str,
    auto_group_k: usize,
    max_metrics: usize,
    format: SuggestProfileFormat,
) -> String {
    match format {
        SuggestProfileFormat::Toml => {
            build_suggested_profile_toml(report, profile_name, auto_group_k, max_metrics)
        }
        SuggestProfileFormat::Json => {
            build_suggested_profile_json(report, profile_name, auto_group_k, max_metrics)
        }
    }
}

fn suggest_profile_format_extension(format: SuggestProfileFormat) -> &'static str {
    match format {
        SuggestProfileFormat::Toml => "toml",
        SuggestProfileFormat::Json => "json",
    }
}

fn suggest_profile_format_label(format: SuggestProfileFormat) -> &'static str {
    match format {
        SuggestProfileFormat::Toml => "TOML",
        SuggestProfileFormat::Json => "JSON",
    }
}

fn default_suggest_profile_path(report_out: &Path, format: SuggestProfileFormat) -> PathBuf {
    report_out.with_extension(suggest_profile_format_extension(format))
}

fn suggest_report_markdown(report: &AnalyzeSuggestReport, profile_path: &Path) -> String {
    let mut md = String::new();
    md.push_str("# Analyze Suggest Report\n\n");
    md.push_str(&format!("- Input: {}\n", report.input));
    md.push_str(&format!("- Sampled rows: {}\n", report.sampled_rows));
    md.push_str(&format!(
        "- Sample mode: {} (seed={})\n",
        report.sample_mode, report.sample_seed
    ));
    md.push_str(&format!(
        "- Suggested profile name: `{}`\n",
        report.profile_name
    ));
    md.push_str(&format!(
        "- Suggested profile path: {}\n\n",
        profile_path.display()
    ));

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
    md.push_str(
        "| Column | Role | Fill % | Distinct | Numeric Ratio | Date Ratio | Top Values |\n",
    );
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
    let base: serde_json::Value = serde_json::from_str(&base_txt)
        .map_err(|e| anyhow!("failed to parse base json '{}': {}", args.base.display(), e))?;
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
            println!(
                "Comparison report (markdown) written to {}",
                out_path.display()
            );
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
            println!(
                "Comparison report (markdown) written to {}",
                out_path.display()
            );
            println!(
                "Comparison report (json) written to {}",
                json_path.display()
            );
        }
    }
    Ok(())
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum InvestigateInputKind {
    JsonArtifacts,
    CsvDatasets,
}

#[derive(Debug, Clone, Serialize)]
struct InvestigationMover {
    segment: String,
    base_records: u64,
    new_records: u64,
    base_share_pct: f64,
    new_share_pct: f64,
    delta_share_pp: f64,
    base_primary_metric_value: f64,
    new_primary_metric_value: f64,
    delta_primary_metric_value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct InvestigationStep {
    depth: usize,
    dimension: String,
    scope: Vec<(String, String)>,
    primary_metric: String,
    base_records: u64,
    new_records: u64,
    segment_count: usize,
    top5_concentration_base_pct: f64,
    top5_concentration_new_pct: f64,
    top5_concentration_delta_pp: f64,
    top1_concentration_base_pct: f64,
    top1_concentration_new_pct: f64,
    top1_concentration_delta_pp: f64,
    movers: Vec<InvestigationMover>,
}

#[derive(Debug, Clone, Serialize)]
struct InvestigationTraceStep {
    depth: usize,
    action: String,
    decision: String,
    scope: Vec<(String, String)>,
    stopping_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct InvestigationMajorChange {
    dimension: String,
    segment: String,
    primary_metric: String,
    delta_primary_metric_value: f64,
    delta_share_pp: f64,
    score: f64,
}

#[derive(Debug, Clone, Serialize)]
struct InvestigationCoverageStep {
    step_index: usize,
    depth: usize,
    scope: Vec<(String, String)>,
    dimension: String,
    strongest_segment: String,
    strongest_delta_primary_metric_value: f64,
    strongest_delta_share_pp: f64,
    strongest_explained_pct_of_total_delta: f64,
    residual_delta_abs_after_step: f64,
}

#[derive(Debug, Clone, Serialize)]
struct InvestigationCoverage {
    total_delta_abs: f64,
    top_level_total_delta: f64,
    top_level_strongest_segment: Option<String>,
    top_level_strongest_delta_abs: f64,
    top_level_strongest_explained_pct: f64,
    step_coverage: Vec<InvestigationCoverageStep>,
}

#[derive(Debug, Clone, Serialize)]
struct InvestigationBranchNode {
    id: String,
    step_index: usize,
    depth: usize,
    scope: Vec<(String, String)>,
    dimension: String,
    primary_metric: String,
    strongest_segment: Option<String>,
    strongest_delta_primary_metric_value: f64,
    strongest_delta_share_pp: f64,
    score: f64,
}

#[derive(Debug, Clone, Serialize)]
struct InvestigationBranchEdge {
    parent_id: String,
    child_id: String,
    score_improvement: f64,
}

#[derive(Debug, Clone, Serialize)]
struct InvestigationBranchGraph {
    nodes: Vec<InvestigationBranchNode>,
    edges: Vec<InvestigationBranchEdge>,
}

#[derive(Debug, Deserialize)]
struct LlmPlannerAction {
    action: String,
    reason: Option<String>,
    params: Option<LlmPlannerParams>,
}

#[derive(Debug, Deserialize)]
struct LlmPlannerParams {
    metric: Option<String>,
    group_by: Option<Vec<String>>,
    filters: Option<HashMap<String, String>>,
}

enum InvestigationExecAction {
    AnalyzeCompare {
        group_by: String,
        scope: Vec<(String, String)>,
        reason: String,
    },
    DrillDown {
        group_by: String,
        scope: Vec<(String, String)>,
        reason: String,
    },
    Stop {
        reason: String,
    },
}

struct ResolvedInvestigateInputs {
    base: PathBuf,
    new: PathBuf,
    base_label: String,
    new_label: String,
}

fn resolve_investigate_inputs(args: &InvestigateWorkflowArgs) -> Result<ResolvedInvestigateInputs> {
    let has_pair = args.base.is_some() || args.new.is_some();
    let has_query =
        args.query.is_some() || args.query_file.is_some() || args.postgres_url.is_some();
    if has_pair && has_query {
        return Err(anyhow!(
            "choose one input mode: (--base and --new) OR (--postgres-url + --query/--query-file)"
        ));
    }

    if has_pair {
        let base = args
            .base
            .clone()
            .ok_or_else(|| anyhow!("--base is required when using file input mode"))?;
        let new = args
            .new
            .clone()
            .ok_or_else(|| anyhow!("--new is required when using file input mode"))?;
        return Ok(ResolvedInvestigateInputs {
            base_label: base.display().to_string(),
            new_label: new.display().to_string(),
            base,
            new,
        });
    }

    let postgres_url = args
        .postgres_url
        .clone()
        .or_else(|| std::env::var("DATABASE_URL").ok())
        .ok_or_else(|| {
            anyhow!(
                "missing input: provide --base/--new OR --postgres-url (or DATABASE_URL) + --query/--query-file"
            )
        })?;
    let query = match (&args.query, &args.query_file) {
        (Some(q), None) => q.clone(),
        (None, Some(path)) => fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read query file '{}': {}", path.display(), e))?,
        (Some(_), Some(_)) => {
            return Err(anyhow!("use only one of --query or --query-file"));
        }
        (None, None) => {
            return Err(anyhow!(
                "postgres investigate requires one of --query or --query-file"
            ));
        }
    };

    let raw_csv = postgres_query_to_temp_csv(
        &postgres_url,
        &query,
        args.postgres_ssl_mode,
        args.postgres_ca_file.as_ref(),
    )?;
    let (headers, sample_rows) = read_csv_headers_and_sample(&raw_csv, 2000)?;
    let date_col = if let Some(dc) = &args.date_column {
        resolve_group_name(dc, &headers)?
    } else {
        auto_detect_date_column(&headers, &sample_rows).ok_or_else(|| {
            anyhow!(
                "failed to auto-detect date column from postgres query output; pass --date-column"
            )
        })?
    };
    let period_cfg = parse_period_cfg_from_investigate_workflow(args, date_col.clone())?;
    let (base_csv, new_csv, base_rows, new_rows) =
        split_csv_into_period_windows(&raw_csv, &period_cfg)?;
    if base_rows == 0 || new_rows == 0 {
        return Err(anyhow!(
            "period windows contain no comparable rows from postgres query (previous={}, current={})",
            base_rows,
            new_rows
        ));
    }

    Ok(ResolvedInvestigateInputs {
        base: base_csv,
        new: new_csv,
        base_label: format!(
            "postgres query ({}) previous window {}..{}",
            period_cfg.date_column, period_cfg.previous_start, period_cfg.previous_end
        ),
        new_label: format!(
            "postgres query ({}) current window {}..{}",
            period_cfg.date_column, period_cfg.current_start, period_cfg.current_end
        ),
    })
}

fn parse_period_cfg_from_investigate_workflow(
    args: &InvestigateWorkflowArgs,
    date_column: String,
) -> Result<PeriodCompareConfig> {
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
            let cs = parse_date_arg(args.current_start.as_deref(), "--current-start is required")?;
            let ce = parse_date_arg(args.current_end.as_deref(), "--current-end is required")?;
            let ps = parse_date_arg(
                args.previous_start.as_deref(),
                "--previous-start is required",
            )?;
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

fn split_csv_into_period_windows(
    input_csv: &Path,
    cfg: &PeriodCompareConfig,
) -> Result<(PathBuf, PathBuf, usize, usize)> {
    let mut rdr = csv::Reader::from_path(input_csv)
        .map_err(|e| anyhow!("failed to read csv '{}': {}", input_csv.display(), e))?;
    let headers = rdr
        .headers()
        .map_err(|e| {
            anyhow!(
                "failed to read headers from '{}': {}",
                input_csv.display(),
                e
            )
        })?
        .clone();
    let date_col = resolve_group_name(&cfg.date_column, &headers)?;
    let date_idx = headers
        .iter()
        .position(|h| h == date_col)
        .ok_or_else(|| anyhow!("date column '{}' not found in query output", date_col))?;

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let base_out = std::env::temp_dir().join(format!("factorlens_investigate_base_{}.csv", ts));
    let new_out = std::env::temp_dir().join(format!("factorlens_investigate_new_{}.csv", ts));
    let mut base_w = csv::Writer::from_path(&base_out)
        .map_err(|e| anyhow!("failed to create '{}': {}", base_out.display(), e))?;
    let mut new_w = csv::Writer::from_path(&new_out)
        .map_err(|e| anyhow!("failed to create '{}': {}", new_out.display(), e))?;
    base_w.write_record(&headers)?;
    new_w.write_record(&headers)?;

    let mut prev_count = 0usize;
    let mut curr_count = 0usize;
    for rec in rdr.records() {
        let rec = rec?;
        let Some(d) = parse_date_like(rec.get(date_idx).unwrap_or("").trim()) else {
            continue;
        };
        if d >= cfg.previous_start && d <= cfg.previous_end {
            base_w.write_record(&rec)?;
            prev_count += 1;
        } else if d >= cfg.current_start && d <= cfg.current_end {
            new_w.write_record(&rec)?;
            curr_count += 1;
        }
    }
    base_w.flush()?;
    new_w.flush()?;
    Ok((base_out, new_out, prev_count, curr_count))
}

fn run_investigate_workflow(args: InvestigateWorkflowArgs) -> Result<()> {
    let args = apply_investigate_config(args)?;
    if args.question.trim().is_empty() {
        return Err(anyhow!(
            "investigate requires --question (or set question in --config/--profile)"
        ));
    }
    if args.planner == InvestigationPlanner::Llm {
        return run_investigate_workflow_llm(args);
    }

    if args.max_depth < 1 {
        return Err(anyhow!("--max-depth must be >= 1"));
    }
    if args.max_branches < 1 {
        return Err(anyhow!("--max-branches must be >= 1"));
    }
    if args.top_movers < 1 {
        return Err(anyhow!("--top-movers must be >= 1"));
    }
    if args.min_delta_abs < 0.0 {
        return Err(anyhow!("--min-delta-abs must be >= 0"));
    }
    if args.min_score_improvement < 0.0 {
        return Err(anyhow!("--min-score-improvement must be >= 0"));
    }
    let planned_branch_nodes = args
        .max_branches
        .saturating_pow(args.max_depth.saturating_sub(1) as u32);
    if planned_branch_nodes > 256 {
        return Err(anyhow!(
            "investigation search is too large (max_branches^depth={}): reduce --max-branches or --max-depth",
            planned_branch_nodes
        ));
    }

    let mode = resolve_investigation_mode(&args);
    let resolved_inputs = resolve_investigate_inputs(&args)?;
    let base_path = &resolved_inputs.base;
    let new_path = &resolved_inputs.new;
    let input_kind = detect_investigate_input_kind(base_path, new_path)?;
    let mode_label = investigation_mode_label(mode);

    let base_json_artifact = if input_kind == InvestigateInputKind::JsonArtifacts {
        Some(read_json_file(base_path, "base")?)
    } else {
        None
    };
    let new_json_artifact = if input_kind == InvestigateInputKind::JsonArtifacts {
        Some(read_json_file(new_path, "new")?)
    } else {
        None
    };
    let input_refs = InvestigationInputRefs {
        base_path,
        new_path,
        input_kind,
        base_artifact: base_json_artifact.as_ref(),
        new_artifact: new_json_artifact.as_ref(),
    };

    let (top_dimension, drill_dimensions) = resolve_investigation_dimensions(
        &args,
        base_path,
        new_path,
        input_kind,
        base_json_artifact.as_ref(),
        new_json_artifact.as_ref(),
    )?;
    let available_dimensions = investigate_available_dimensions(&top_dimension, &drill_dimensions);
    let major_global_changes = identify_major_global_changes(
        &args,
        &input_refs,
        mode,
        &available_dimensions,
        major_change_limit_for_dimensions(&available_dimensions),
    )?;

    if args.verbose {
        println!("[route] Question classified as {}", mode_label);
        println!(
            "[config] input_mode={} top_dimension={} max_depth={} max_branches={}",
            investigate_input_kind_label(input_kind),
            top_dimension,
            args.max_depth,
            args.max_branches
        );
    }

    let mut steps = Vec::<InvestigationStep>::new();
    let mut trace = Vec::<InvestigationTraceStep>::new();
    let mut stop_reason: Option<String> = None;
    let root_scope = Vec::<(String, String)>::new();
    let root_step =
        investigation_step_from_inputs(&args, &input_refs, &top_dimension, &root_scope, mode)?;
    if args.verbose {
        println!("[step 0] compared on dimension '{}'", top_dimension);
    }
    trace.push(InvestigationTraceStep {
        depth: 0,
        action: "top_level_compare".to_string(),
        decision: format!("ran base/new comparison grouped by {}", top_dimension),
        scope: root_scope.clone(),
        stopping_reason: None,
    });
    steps.push(root_step.clone());
    let mut frontier = vec![root_step];

    for depth in 1..args.max_depth {
        if input_kind == InvestigateInputKind::JsonArtifacts {
            stop_reason = Some(
                "stopped after top-level analysis: drill-down requires CSV datasets, but --base/--new are JSON artifacts"
                    .to_string(),
            );
            break;
        }

        let mut next_frontier = Vec::<InvestigationStep>::new();
        let mut expanded = false;
        let mut saw_next_dimension = false;
        let mut saw_eligible_candidate = false;
        let mut saw_composite_segment = false;
        let mut saw_low_delta_abs = false;
        let mut saw_low_improvement = false;

        for parent_step in &frontier {
            let used_dims = used_dimensions_for_step(parent_step);
            let remaining_dimensions = remaining_drill_dimensions(&drill_dimensions, &used_dims);
            if remaining_dimensions.is_empty() {
                continue;
            }
            saw_next_dimension = true;

            let had_base_candidates = parent_step.movers.iter().any(|m| {
                drill_candidate_matches_base_thresholds(
                    m,
                    mode,
                    args.min_contribution,
                    args.min_slice_rows,
                )
            });
            let candidates = select_drill_candidates(
                parent_step,
                mode,
                args.min_contribution,
                args.min_delta_abs,
                args.min_slice_rows,
                args.max_branches,
            );
            if candidates.is_empty() {
                if mode != InvestigationMode::ConcentrationDrivers
                    && args.min_delta_abs > 0.0
                    && had_base_candidates
                {
                    saw_low_delta_abs = true;
                }
                continue;
            }
            saw_eligible_candidate = true;

            for mover in candidates {
                if mover.segment.contains(" | ") {
                    saw_composite_segment = true;
                    continue;
                }

                let mut child_scope = parent_step.scope.clone();
                child_scope.push((parent_step.dimension.clone(), mover.segment.clone()));
                let parent_score = mover_score(mover, mode);
                let mut best_next: Option<(String, InvestigationStep, f64, f64, f64)> = None;
                for next_dimension in &remaining_dimensions {
                    let step = investigation_step_from_inputs(
                        &args,
                        &input_refs,
                        next_dimension,
                        &child_scope,
                        mode,
                    )?;
                    let score = step
                        .movers
                        .first()
                        .map(|m| mover_score(m, mode))
                        .unwrap_or(0.0);
                    let delta_abs = top_step_delta_abs(&step);
                    let gain = score - parent_score;
                    let replace = match &best_next {
                        None => true,
                        Some((best_dim, _, best_score, best_gain, best_delta_abs)) => {
                            gain > *best_gain + f64::EPSILON
                                || ((gain - *best_gain).abs() <= f64::EPSILON
                                    && (delta_abs > *best_delta_abs + f64::EPSILON
                                        || ((delta_abs - *best_delta_abs).abs() <= f64::EPSILON
                                            && (score > *best_score + f64::EPSILON
                                                || ((score - *best_score).abs() <= f64::EPSILON
                                                    && next_dimension < best_dim)))))
                                || ((gain - *best_gain).abs() <= f64::EPSILON
                                    && (delta_abs - *best_delta_abs).abs() <= f64::EPSILON
                                    && (score - *best_score).abs() <= f64::EPSILON
                                    && next_dimension < best_dim)
                        }
                    };
                    if replace {
                        best_next = Some((next_dimension.clone(), step, score, gain, delta_abs));
                    }
                }
                let Some((next_dimension, child_step, _, _, child_delta_abs)) = best_next else {
                    continue;
                };
                if mode != InvestigationMode::ConcentrationDrivers
                    && args.min_delta_abs > 0.0
                    && child_delta_abs + f64::EPSILON < args.min_delta_abs
                {
                    saw_low_delta_abs = true;
                    continue;
                }
                let child_score = top_step_score(&child_step, mode);
                if args.min_score_improvement > 0.0
                    && child_score + f64::EPSILON < parent_score + args.min_score_improvement
                {
                    saw_low_improvement = true;
                    continue;
                }

                let decision =
                    deterministic_drill_decision(parent_step, mover, mode, &next_dimension);
                if args.verbose {
                    println!("[decision] {}", decision);
                }
                if args.verbose {
                    println!(
                        "[step {}] scope={} group_by={}",
                        depth,
                        child_scope
                            .iter()
                            .map(|(k, v)| format!("{}={}", k, v))
                            .collect::<Vec<_>>()
                            .join(", "),
                        next_dimension
                    );
                }
                trace.push(InvestigationTraceStep {
                    depth,
                    action: "drill_down".to_string(),
                    decision,
                    scope: child_scope.clone(),
                    stopping_reason: None,
                });
                steps.push(child_step.clone());
                next_frontier.push(child_step);
                expanded = true;
            }
        }

        if !expanded {
            if !saw_next_dimension {
                stop_reason = Some("no remaining drill dimension was available".to_string());
            } else if saw_low_delta_abs {
                stop_reason = Some(format!(
                    "no drill candidate met absolute-delta threshold (min_delta_abs={})",
                    fmt_num(args.min_delta_abs, 2)
                ));
            } else if saw_low_improvement {
                stop_reason = Some(format!(
                    "no drill candidate met score-improvement threshold (min_score_improvement={})",
                    fmt_num(args.min_score_improvement, 2)
                ));
            } else if !saw_eligible_candidate {
                stop_reason = Some(format!(
                    "no drill candidate met thresholds (min_contribution={}, min_slice_rows={})",
                    fmt_num(args.min_contribution, 2),
                    args.min_slice_rows
                ));
            } else if saw_composite_segment {
                stop_reason = Some(
                    "top drill candidates were composite groups; use one dimension per step in v1"
                        .to_string(),
                );
            } else {
                stop_reason = Some("no drill-down branch could be expanded".to_string());
            }
            break;
        }
        frontier = next_frontier;
    }

    if stop_reason.is_none() {
        stop_reason = Some(format!("reached max depth {}", args.max_depth));
    }
    if let Some(last) = trace.last_mut() {
        last.stopping_reason = stop_reason.clone();
    }
    if args.verbose {
        println!("[stop] {}", stop_reason.clone().unwrap_or_default());
    }

    let recommended_next_question = recommended_next_question(mode, steps.last());
    let coverage = build_investigation_coverage(&steps);
    let branch_graph = build_investigation_branch_graph(&steps, mode);
    let input_labels = InvestigationInputLabels {
        base: &resolved_inputs.base_label,
        new: &resolved_inputs.new_label,
    };
    let markdown = render_investigation_workflow_markdown(
        &args,
        &input_labels,
        mode,
        &InvestigationRenderData {
            major_global_changes: &major_global_changes,
            coverage: &coverage,
            branch_graph: &branch_graph,
            steps: &steps,
            trace: &trace,
            stop_reason: stop_reason.as_deref().unwrap_or("stopped"),
            recommended_next_question: &recommended_next_question,
        },
    );
    let json_out = serde_json::json!({
        "question": args.question,
        "mode": mode_label,
        "input": {
            "kind": investigate_input_kind_label(input_kind),
            "base": resolved_inputs.base_label,
            "new": resolved_inputs.new_label
        },
        "config": {
            "dimensions": args.dimensions,
            "drill_fields": args.drill_fields,
            "max_depth": args.max_depth,
            "max_branches": args.max_branches,
            "min_contribution": args.min_contribution,
            "min_delta_abs": args.min_delta_abs,
            "min_score_improvement": args.min_score_improvement,
            "min_slice_rows": args.min_slice_rows,
            "top_movers": args.top_movers,
            "metric": args.metric
        },
        "steps": steps,
        "major_global_changes": major_global_changes,
        "coverage": coverage,
        "branch_graph": branch_graph,
        "trace": trace,
        "stopping_reason": stop_reason,
        "recommended_next_question": recommended_next_question
    });

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| default_investigate_workflow_out(&args));
    ensure_parent_dir(&out_path)?;
    match args.output_format {
        InvestigateOutputFormat::Md => {
            fs::write(&out_path, markdown)?;
            println!(
                "Investigate report (markdown) written to {}",
                out_path.display()
            );
        }
        InvestigateOutputFormat::Json => {
            fs::write(&out_path, serde_json::to_string_pretty(&json_out)?)?;
            println!(
                "Investigate report (json) written to {}",
                out_path.display()
            );
        }
        InvestigateOutputFormat::Both => {
            let (md_path, json_path) = investigate_both_paths(&out_path);
            ensure_parent_dir(&md_path)?;
            ensure_parent_dir(&json_path)?;
            fs::write(&md_path, markdown)?;
            fs::write(&json_path, serde_json::to_string_pretty(&json_out)?)?;
            println!("Investigate report written to {}", md_path.display());
            println!("Investigate JSON written to {}", json_path.display());
        }
    }

    Ok(())
}

fn run_investigate_workflow_llm(args: InvestigateWorkflowArgs) -> Result<()> {
    if args.max_depth < 1 {
        return Err(anyhow!("--max-depth must be >= 1"));
    }
    if args.max_branches < 1 {
        return Err(anyhow!("--max-branches must be >= 1"));
    }
    if args.top_movers < 1 {
        return Err(anyhow!("--top-movers must be >= 1"));
    }
    if args.min_delta_abs < 0.0 {
        return Err(anyhow!("--min-delta-abs must be >= 0"));
    }
    if args.min_score_improvement < 0.0 {
        return Err(anyhow!("--min-score-improvement must be >= 0"));
    }

    let trace_enabled = args.verbose || args.trace;
    let mode = resolve_investigation_mode(&args);
    let resolved_inputs = resolve_investigate_inputs(&args)?;
    let base_path = &resolved_inputs.base;
    let new_path = &resolved_inputs.new;
    let input_kind = detect_investigate_input_kind(base_path, new_path)?;

    let base_json_artifact = if input_kind == InvestigateInputKind::JsonArtifacts {
        Some(read_json_file(base_path, "base")?)
    } else {
        None
    };
    let new_json_artifact = if input_kind == InvestigateInputKind::JsonArtifacts {
        Some(read_json_file(new_path, "new")?)
    } else {
        None
    };
    let input_refs = InvestigationInputRefs {
        base_path,
        new_path,
        input_kind,
        base_artifact: base_json_artifact.as_ref(),
        new_artifact: new_json_artifact.as_ref(),
    };

    let (top_dimension, drill_dimensions) = resolve_investigation_dimensions(
        &args,
        base_path,
        new_path,
        input_kind,
        base_json_artifact.as_ref(),
        new_json_artifact.as_ref(),
    )?;
    let available_dimensions = investigate_available_dimensions(&top_dimension, &drill_dimensions);
    let major_global_changes = identify_major_global_changes(
        &args,
        &input_refs,
        mode,
        &available_dimensions,
        major_change_limit_for_dimensions(&available_dimensions),
    )?;

    let planner_backend = match args.planner_backend {
        BackendArg::Local => Backend::Local,
        BackendArg::Bedrock => Backend::Bedrock,
    };
    let planner_model = resolve_planner_model(&args)?;
    let planner = build_client(planner_backend, planner_model);
    let local_fallback_planner = build_local_fallback_planner(&args);

    if trace_enabled {
        println!(
            "[planner] llm backend={} input_mode={} dimensions={}",
            planner_backend_label(args.planner_backend),
            investigate_input_kind_label(input_kind),
            available_dimensions.join(",")
        );
        if let Some(model) = local_fallback_planner_model_name(&args) {
            println!("[planner] local fallback enabled model={}", model);
        }
    }

    let mut steps = Vec::<InvestigationStep>::new();
    let mut trace = Vec::<InvestigationTraceStep>::new();
    let mut stop_reason: Option<String> = None;

    for depth in 0..args.max_depth {
        let planned = llm_plan_next_action_with_fallback(
            planner.as_ref(),
            local_fallback_planner.as_deref(),
            &args,
            &available_dimensions,
            &steps,
            depth,
            input_kind,
        );
        let exec_action = match planned {
            Ok((action, source)) => {
                if trace_enabled && source == "local_fallback" {
                    println!(
                        "[planner] primary backend failed at depth={}, local fallback produced action",
                        depth
                    );
                }
                action
            }
            Err(err) => {
                let err_text = compact_error_message(&err.to_string());
                let fallback = deterministic_fallback_action(
                    &args,
                    mode,
                    input_kind,
                    &top_dimension,
                    &drill_dimensions,
                    &steps,
                );
                if trace_enabled {
                    println!("[planner] fallback due to invalid LLM action: {}", err_text);
                }
                match fallback {
                    InvestigationExecAction::AnalyzeCompare {
                        group_by,
                        scope,
                        reason,
                    } => InvestigationExecAction::AnalyzeCompare {
                        group_by,
                        scope,
                        reason: format!("{} (fallback: {})", reason, err_text),
                    },
                    InvestigationExecAction::DrillDown {
                        group_by,
                        scope,
                        reason,
                    } => InvestigationExecAction::DrillDown {
                        group_by,
                        scope,
                        reason: format!("{} (fallback: {})", reason, err_text),
                    },
                    InvestigationExecAction::Stop { reason } => InvestigationExecAction::Stop {
                        reason: format!("{} (fallback: {})", reason, err_text),
                    },
                }
            }
        };

        match exec_action {
            InvestigationExecAction::Stop { reason } => {
                stop_reason = Some(reason.clone());
                trace.push(InvestigationTraceStep {
                    depth,
                    action: "stop".to_string(),
                    decision: reason,
                    scope: steps.last().map(|s| s.scope.clone()).unwrap_or_default(),
                    stopping_reason: stop_reason.clone(),
                });
                break;
            }
            InvestigationExecAction::AnalyzeCompare {
                group_by,
                scope,
                reason,
            } => {
                let step =
                    investigation_step_from_inputs(&args, &input_refs, &group_by, &scope, mode)?;
                if mode != InvestigationMode::ConcentrationDrivers && args.min_delta_abs > 0.0 {
                    let current_delta_abs = top_step_delta_abs(&step);
                    if current_delta_abs + f64::EPSILON < args.min_delta_abs {
                        stop_reason = Some(format!(
                            "no planned step met absolute-delta threshold (min_delta_abs={})",
                            fmt_num(args.min_delta_abs, 2)
                        ));
                        trace.push(InvestigationTraceStep {
                            depth,
                            action: "stop".to_string(),
                            decision: format!(
                                "planned analyze_compare did not meet absolute delta threshold (current_abs={}, required={})",
                                fmt_num(current_delta_abs, 2),
                                fmt_num(args.min_delta_abs, 2)
                            ),
                            scope,
                            stopping_reason: stop_reason.clone(),
                        });
                        break;
                    }
                }
                if args.min_score_improvement > 0.0 {
                    if let Some(prev_step) = steps.last() {
                        let prev_score = top_step_score(prev_step, mode);
                        let current_score = top_step_score(&step, mode);
                        if current_score + f64::EPSILON < prev_score + args.min_score_improvement {
                            stop_reason = Some(format!(
                                "no planned step met score-improvement threshold (min_score_improvement={})",
                                fmt_num(args.min_score_improvement, 2)
                            ));
                            trace.push(InvestigationTraceStep {
                                depth,
                                action: "stop".to_string(),
                                decision: format!(
                                    "planned analyze_compare did not improve score enough (prev={}, current={}, required +{})",
                                    fmt_num(prev_score, 2),
                                    fmt_num(current_score, 2),
                                    fmt_num(args.min_score_improvement, 2)
                                ),
                                scope,
                                stopping_reason: stop_reason.clone(),
                            });
                            break;
                        }
                    }
                }
                if trace_enabled {
                    println!(
                        "[planner] analyze_compare depth={} group_by={} scope={} reason={}",
                        depth,
                        group_by,
                        format_scope(&scope),
                        reason
                    );
                }
                trace.push(InvestigationTraceStep {
                    depth,
                    action: "analyze_compare".to_string(),
                    decision: reason,
                    scope,
                    stopping_reason: None,
                });
                steps.push(step);
            }
            InvestigationExecAction::DrillDown {
                group_by,
                scope,
                reason,
            } => {
                let step =
                    investigation_step_from_inputs(&args, &input_refs, &group_by, &scope, mode)?;
                if mode != InvestigationMode::ConcentrationDrivers && args.min_delta_abs > 0.0 {
                    let current_delta_abs = top_step_delta_abs(&step);
                    if current_delta_abs + f64::EPSILON < args.min_delta_abs {
                        stop_reason = Some(format!(
                            "no planned step met absolute-delta threshold (min_delta_abs={})",
                            fmt_num(args.min_delta_abs, 2)
                        ));
                        trace.push(InvestigationTraceStep {
                            depth,
                            action: "stop".to_string(),
                            decision: format!(
                                "planned drill_down did not meet absolute delta threshold (current_abs={}, required={})",
                                fmt_num(current_delta_abs, 2),
                                fmt_num(args.min_delta_abs, 2)
                            ),
                            scope,
                            stopping_reason: stop_reason.clone(),
                        });
                        break;
                    }
                }
                if args.min_score_improvement > 0.0 {
                    if let Some(prev_step) = steps.last() {
                        let prev_score = top_step_score(prev_step, mode);
                        let current_score = top_step_score(&step, mode);
                        if current_score + f64::EPSILON < prev_score + args.min_score_improvement {
                            stop_reason = Some(format!(
                                "no planned step met score-improvement threshold (min_score_improvement={})",
                                fmt_num(args.min_score_improvement, 2)
                            ));
                            trace.push(InvestigationTraceStep {
                                depth,
                                action: "stop".to_string(),
                                decision: format!(
                                    "planned drill_down did not improve score enough (prev={}, current={}, required +{})",
                                    fmt_num(prev_score, 2),
                                    fmt_num(current_score, 2),
                                    fmt_num(args.min_score_improvement, 2)
                                ),
                                scope,
                                stopping_reason: stop_reason.clone(),
                            });
                            break;
                        }
                    }
                }
                if trace_enabled {
                    println!(
                        "[planner] drill_down depth={} group_by={} scope={} reason={}",
                        depth,
                        group_by,
                        format_scope(&scope),
                        reason
                    );
                }
                trace.push(InvestigationTraceStep {
                    depth,
                    action: "drill_down".to_string(),
                    decision: reason,
                    scope,
                    stopping_reason: None,
                });
                steps.push(step);
            }
        }
    }

    if stop_reason.is_none() {
        stop_reason = Some(format!("reached max depth {}", args.max_depth));
    }
    if let Some(last) = trace.last_mut() {
        last.stopping_reason = stop_reason.clone();
    }

    let recommended_next_question = recommended_next_question(mode, steps.last());
    let coverage = build_investigation_coverage(&steps);
    let branch_graph = build_investigation_branch_graph(&steps, mode);
    let llm_summary = llm_finalize_summary_with_fallback(
        planner.as_ref(),
        local_fallback_planner.as_deref(),
        &args.question,
        &steps,
        &trace,
        stop_reason.as_deref().unwrap_or("stopped"),
    )
    .unwrap_or_else(|_| {
        deterministic_summary_from_steps(
            &steps,
            stop_reason.as_deref().unwrap_or("stopped"),
            &recommended_next_question,
        )
    });

    let mut markdown = render_investigation_workflow_markdown(
        &args,
        &InvestigationInputLabels {
            base: &resolved_inputs.base_label,
            new: &resolved_inputs.new_label,
        },
        mode,
        &InvestigationRenderData {
            major_global_changes: &major_global_changes,
            coverage: &coverage,
            branch_graph: &branch_graph,
            steps: &steps,
            trace: &trace,
            stop_reason: stop_reason.as_deref().unwrap_or("stopped"),
            recommended_next_question: &recommended_next_question,
        },
    );
    markdown.push_str("\n## Final summary\n\n");
    for line in llm_summary.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            markdown.push_str(&format!("- {}\n", trimmed));
        }
    }

    let json_out = serde_json::json!({
        "question": args.question,
        "mode": investigation_mode_label(mode),
        "planner": "llm",
        "planner_backend": planner_backend_label(args.planner_backend),
        "input": {
            "kind": investigate_input_kind_label(input_kind),
            "base": resolved_inputs.base_label,
            "new": resolved_inputs.new_label
        },
        "config": {
            "dimensions": args.dimensions,
            "drill_fields": args.drill_fields,
            "max_depth": args.max_depth,
            "max_branches": args.max_branches,
            "min_contribution": args.min_contribution,
            "min_delta_abs": args.min_delta_abs,
            "min_score_improvement": args.min_score_improvement,
            "min_slice_rows": args.min_slice_rows,
            "top_movers": args.top_movers,
            "metric": args.metric
        },
        "steps": steps,
        "major_global_changes": major_global_changes,
        "coverage": coverage,
        "branch_graph": branch_graph,
        "trace": trace,
        "stopping_reason": stop_reason,
        "recommended_next_question": recommended_next_question,
        "final_summary": llm_summary,
    });

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| default_investigate_workflow_out(&args));
    ensure_parent_dir(&out_path)?;
    match args.output_format {
        InvestigateOutputFormat::Md => {
            fs::write(&out_path, markdown)?;
            println!(
                "Investigate report (markdown) written to {}",
                out_path.display()
            );
        }
        InvestigateOutputFormat::Json => {
            fs::write(&out_path, serde_json::to_string_pretty(&json_out)?)?;
            println!(
                "Investigate report (json) written to {}",
                out_path.display()
            );
        }
        InvestigateOutputFormat::Both => {
            let (md_path, json_path) = investigate_both_paths(&out_path);
            ensure_parent_dir(&md_path)?;
            ensure_parent_dir(&json_path)?;
            fs::write(&md_path, markdown)?;
            fs::write(&json_path, serde_json::to_string_pretty(&json_out)?)?;
            println!("Investigate report written to {}", md_path.display());
            println!("Investigate JSON written to {}", json_path.display());
        }
    }
    Ok(())
}

fn llm_plan_next_action_with_fallback(
    planner: &dyn LlmClient,
    local_fallback: Option<&dyn LlmClient>,
    args: &InvestigateWorkflowArgs,
    available_dimensions: &[String],
    steps: &[InvestigationStep],
    depth: usize,
    input_kind: InvestigateInputKind,
) -> Result<(InvestigationExecAction, &'static str)> {
    match llm_plan_next_action(
        planner,
        args,
        available_dimensions,
        steps,
        depth,
        input_kind,
        "primary",
    ) {
        Ok(action) => Ok((action, "primary")),
        Err(primary_err) => {
            if let Some(local) = local_fallback {
                match llm_plan_next_action(
                    local,
                    args,
                    available_dimensions,
                    steps,
                    depth,
                    input_kind,
                    "local_fallback",
                ) {
                    Ok(action) => Ok((action, "local_fallback")),
                    Err(local_err) => Err(anyhow!(
                        "primary planner failed: {}; local fallback failed: {}",
                        primary_err,
                        local_err
                    )),
                }
            } else {
                Err(primary_err)
            }
        }
    }
}

fn llm_plan_next_action(
    planner: &dyn LlmClient,
    args: &InvestigateWorkflowArgs,
    available_dimensions: &[String],
    steps: &[InvestigationStep],
    depth: usize,
    input_kind: InvestigateInputKind,
    source: &str,
) -> Result<InvestigationExecAction> {
    let history = steps
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let top = s.movers.first();
            serde_json::json!({
                "index": i,
                "depth": s.depth,
                "dimension": s.dimension,
                "scope": s.scope,
                "top_mover": top.map(|m| serde_json::json!({
                    "segment": m.segment,
                    "delta_metric": m.delta_primary_metric_value,
                    "delta_share_pp": m.delta_share_pp,
                }))
            })
        })
        .collect::<Vec<_>>();
    let last_result = steps.last().and_then(|s| s.movers.first()).map(|m| {
        serde_json::json!({
            "dimension": steps.last().map(|x| x.dimension.clone()).unwrap_or_default(),
            "value": m.segment,
            "contribution": m.delta_primary_metric_value.abs(),
            "delta_share_pp": m.delta_share_pp,
        })
    });
    let user_payload = serde_json::json!({
        "question": args.question,
        "available_dimensions": available_dimensions,
        "history": history,
        "last_result": last_result,
        "constraints": {
            "current_depth": depth,
            "max_depth": args.max_depth,
            "input_kind": investigate_input_kind_label(input_kind),
            "min_contribution": args.min_contribution,
            "min_delta_abs": args.min_delta_abs,
            "min_score_improvement": args.min_score_improvement,
            "min_slice_rows": args.min_slice_rows,
            "required_metric": args.metric
        }
    });
    let system_prompt = "You are an analysis planner for FactorLens. Choose exactly one action: analyze_compare, drill_down, or stop. Return only valid JSON. Never invent dimensions or metrics. If required_metric is present, use exactly that metric name or omit params.metric. Do not abbreviate metric names (for example revenue is not revenue_usd). Use previous results and prefer the strongest valid driver.";
    let user_prompt = format!(
        "Return ONLY one JSON object (no markdown, no prose).\nRequired keys:\n- action: analyze_compare | drill_down | stop\n- reason: short string\n- params: object (optional for stop) with optional metric, group_by (array), filters (object)\nRules:\n- Use only available_dimensions.\n- For drill_down include non-empty filters.\n- If required_metric is set, params.metric must be exactly that value (or omitted).\n- Do not echo this prompt.\nInput payload:\n{}",
        serde_json::to_string_pretty(&user_payload)?
    );
    let raw = planner.answer(system_prompt, &user_prompt)?;
    if args.trace || args.verbose {
        println!(
            "[planner] raw {} action={}",
            source,
            compact_trace_preview(&raw, 320)
        );
    }
    let parsed_candidates = parse_llm_planner_actions(&raw)?;
    let mut validation_errors = Vec::<String>::new();
    for candidate in parsed_candidates.iter().rev() {
        match validate_llm_planner_action(candidate, args, available_dimensions, steps, input_kind)
        {
            Ok(valid) => return Ok(valid),
            Err(err) => validation_errors.push(compact_error_message(&err.to_string())),
        }
    }
    let preview = validation_errors
        .into_iter()
        .take(3)
        .collect::<Vec<_>>()
        .join(" | ");
    Err(anyhow!("no valid llm action candidate: {}", preview))
}

#[cfg(test)]
fn parse_llm_planner_action(raw: &str) -> Result<LlmPlannerAction> {
    let mut parsed = parse_llm_planner_actions(raw)?;
    parsed
        .pop()
        .ok_or_else(|| anyhow!("llm output missing parseable JSON object"))
}

fn parse_llm_planner_actions(raw: &str) -> Result<Vec<LlmPlannerAction>> {
    if let Ok(v) = serde_json::from_str::<LlmPlannerAction>(raw.trim()) {
        return Ok(vec![v]);
    }
    let mut parsed = Vec::<LlmPlannerAction>::new();
    let mut last_err: Option<serde_json::Error> = None;
    for (start, ch) in raw.char_indices() {
        if ch != '{' {
            continue;
        }
        let Some(slice) = extract_json_object_from(raw, start) else {
            continue;
        };
        match serde_json::from_str::<LlmPlannerAction>(slice) {
            Ok(v) => parsed.push(v),
            Err(err) => last_err = Some(err),
        }
    }
    if !parsed.is_empty() {
        return Ok(parsed);
    }
    if let Some(err) = last_err {
        return Err(anyhow!("failed to parse llm planner JSON: {}", err));
    }
    Err(anyhow!("llm output missing parseable JSON object"))
}

fn extract_json_object_from(raw: &str, start: usize) -> Option<&str> {
    let bytes = raw.as_bytes();
    if start >= bytes.len() || bytes[start] != b'{' {
        return None;
    }

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in raw[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    let end = start + offset + ch.len_utf8();
                    return Some(&raw[start..end]);
                }
            }
            _ => {}
        }
    }
    None
}

fn validate_llm_planner_action(
    proposed: &LlmPlannerAction,
    args: &InvestigateWorkflowArgs,
    available_dimensions: &[String],
    steps: &[InvestigationStep],
    input_kind: InvestigateInputKind,
) -> Result<InvestigationExecAction> {
    let action = proposed.action.trim().to_ascii_lowercase();
    let reason_raw = proposed
        .reason
        .clone()
        .unwrap_or_else(|| "llm planner decision".to_string());
    let reason = normalize_planner_reason(action.as_str(), &reason_raw);
    let params = proposed.params.as_ref();

    if let Some(metric) = params.and_then(|p| p.metric.as_deref()) {
        match args.metric.as_deref() {
            Some(expected) if metric_matches_expected(metric, expected) => {}
            Some(_expected) => {
                // Ignore planner metric overrides in investigate v1.
                // We always execute with the CLI metric for deterministic grounding.
            }
            None => {
                return Err(anyhow!(
                    "llm requested metric '{}' but metric override is not supported in v1",
                    metric
                ));
            }
        }
    }

    let group_by_raw = params
        .and_then(|p| p.group_by.as_ref())
        .and_then(|xs| xs.first())
        .map(|s| s.as_str())
        .unwrap_or_default();
    let mut scope = ordered_scope_from_filters(
        params.and_then(|p| p.filters.as_ref()),
        available_dimensions,
    )?;

    match action.as_str() {
        "stop" => {
            if steps.is_empty() {
                return Err(anyhow!("first llm action cannot be stop"));
            }
            if input_kind == InvestigateInputKind::CsvDatasets && args.max_depth > 1 {
                if let Some(scope) = infer_scope_from_last_step(steps) {
                    if let Some(group_by) = infer_group_by_from_context(
                        "drill_down",
                        steps,
                        available_dimensions,
                        &scope,
                    ) {
                        if !is_repeated_path(steps, &group_by, &scope) {
                            return Ok(InvestigationExecAction::DrillDown {
                                group_by,
                                scope,
                                reason: "auto-follow drill-down from prior top mover".to_string(),
                            });
                        }
                    }
                }
            }
            Ok(InvestigationExecAction::Stop { reason })
        }
        "analyze_compare" | "drill_down" => {
            let mut drill_mode = action == "drill_down";
            if steps.is_empty() && drill_mode {
                return Err(anyhow!("first llm action must be analyze_compare"));
            }
            let mut group_by = if group_by_raw.trim().is_empty() {
                infer_group_by_from_context(action.as_str(), steps, available_dimensions, &scope)
                    .ok_or_else(|| anyhow!("llm action requires params.group_by[0]"))?
            } else {
                resolve_dimension_name(group_by_raw, available_dimensions)
                    .ok_or_else(|| anyhow!("llm requested unknown dimension '{}'", group_by_raw))?
            };

            if drill_mode {
                if input_kind == InvestigateInputKind::JsonArtifacts {
                    return Err(anyhow!("drill_down is unsupported for JSON artifact mode"));
                }
                if scope.is_empty() {
                    scope = infer_scope_from_last_step(steps)
                        .ok_or_else(|| anyhow!("drill_down requires non-empty params.filters"))?;
                }
            } else if steps.is_empty() {
                scope.clear();
            }

            if let Some(last) = steps.last() {
                if !last.scope.is_empty() && !scope.is_empty() {
                    if let Some((dim, prev, next)) = conflicting_scope_binding(&last.scope, &scope)
                    {
                        return Err(anyhow!(
                            "llm scope conflicts with prior scope on {} ('{}' vs '{}')",
                            dim,
                            prev,
                            next
                        ));
                    }
                    if !scope_is_superset_of(&scope, &last.scope) {
                        scope = merge_scope(&last.scope, &scope);
                    }
                }
            }

            if scope_has_dimension(&scope, &group_by) {
                let adjusted =
                    infer_group_by_from_context("drill_down", steps, available_dimensions, &scope)
                        .or_else(|| {
                            available_dimensions
                                .iter()
                                .find(|d| !scope_has_dimension(&scope, d))
                                .cloned()
                        });
                let Some(next_group_by) = adjusted else {
                    return Err(anyhow!(
                        "llm selected group_by '{}' which is already fixed by scope={}",
                        group_by,
                        format_scope(&scope)
                    ));
                };
                if is_repeated_path(steps, &next_group_by, &scope) {
                    return Err(anyhow!(
                        "llm repeated a previously executed path (dimension='{}' scope={})",
                        next_group_by,
                        format_scope(&scope)
                    ));
                }
                group_by = next_group_by;
            }

            if !drill_mode
                && !steps.is_empty()
                && scope.is_empty()
                && input_kind != InvestigateInputKind::JsonArtifacts
            {
                let inferred_scope = infer_scope_from_last_step(steps).ok_or_else(|| {
                    anyhow!("analyze_compare after first step requires a drillable prior result")
                })?;
                let next_group_by = infer_group_by_from_context(
                    "drill_down",
                    steps,
                    available_dimensions,
                    &inferred_scope,
                )
                .ok_or_else(|| anyhow!("could not infer next drill dimension"))?;
                if is_repeated_path(steps, &next_group_by, &inferred_scope) {
                    return Err(anyhow!(
                        "llm repeated a previously executed path (dimension='{}' scope={})",
                        next_group_by,
                        format_scope(&inferred_scope)
                    ));
                }
                drill_mode = true;
                scope = inferred_scope;
                group_by = next_group_by;
            }

            if is_repeated_path(steps, &group_by, &scope) {
                // Local models often repeat the previous top-level path. Auto-pivot to a
                // valid drill-down path when possible instead of hard-failing.
                if !drill_mode && input_kind != InvestigateInputKind::JsonArtifacts {
                    if let Some(inferred_scope) = infer_scope_from_last_step(steps) {
                        if let Some(next_group_by) = infer_group_by_from_context(
                            "drill_down",
                            steps,
                            available_dimensions,
                            &inferred_scope,
                        ) {
                            if !is_repeated_path(steps, &next_group_by, &inferred_scope) {
                                drill_mode = true;
                                scope = inferred_scope;
                                group_by = next_group_by;
                            }
                        }
                    }
                }
            }

            if is_repeated_path(steps, &group_by, &scope) {
                let adjusted = infer_group_by_from_context(
                    action.as_str(),
                    steps,
                    available_dimensions,
                    &scope,
                );
                if let Some(next_group_by) = adjusted {
                    if !next_group_by.eq_ignore_ascii_case(&group_by)
                        && !is_repeated_path(steps, &next_group_by, &scope)
                    {
                        group_by = next_group_by;
                    } else {
                        return Err(anyhow!(
                            "llm repeated a previously executed path (dimension='{}' scope={})",
                            group_by,
                            format_scope(&scope)
                        ));
                    }
                } else {
                    return Err(anyhow!(
                        "llm repeated a previously executed path (dimension='{}' scope={})",
                        group_by,
                        format_scope(&scope)
                    ));
                }
            }

            if drill_mode {
                Ok(InvestigationExecAction::DrillDown {
                    group_by,
                    scope,
                    reason,
                })
            } else {
                Ok(InvestigationExecAction::AnalyzeCompare {
                    group_by,
                    scope,
                    reason,
                })
            }
        }
        _ => Err(anyhow!("invalid llm action '{}'", proposed.action)),
    }
}

fn normalize_planner_reason(action: &str, raw_reason: &str) -> String {
    let trimmed = raw_reason.trim();
    if trimmed.is_empty() {
        return default_planner_reason(action).to_string();
    }

    let collapsed = trimmed
        .replace('_', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let lower = collapsed.to_ascii_lowercase();
    if collapsed.len() < 8 || planner_reason_is_low_signal(&lower) {
        return default_planner_reason(action).to_string();
    }

    collapsed
}

fn planner_reason_is_low_signal(lower: &str) -> bool {
    if matches!(
        lower,
        "previous result" | "previous results" | "previousresult" | "n/a" | "none" | "unknown"
    ) {
        return true;
    }

    lower.starts_with("previous result")
        || lower.starts_with("previous results")
        || lower.contains("not strong enough")
        || lower.contains("too shallow")
        || lower.contains("at max depth")
        || lower.contains("results are weak")
        || lower.contains("result is weak")
        || lower.contains("result is null")
        || lower.contains("results are empty")
        || lower.contains("no strong drivers")
        || lower == "strongest valid driver"
}

fn default_planner_reason(action: &str) -> &'static str {
    match action {
        "analyze_compare" => "planner selected top-level comparison",
        "drill_down" => "planner selected drill-down from prior top mover",
        "stop" => "planner selected stop",
        _ => "planner decision",
    }
}

fn is_repeated_path(
    steps: &[InvestigationStep],
    group_by: &str,
    scope: &[(String, String)],
) -> bool {
    steps
        .iter()
        .any(|s| s.dimension.eq_ignore_ascii_case(group_by) && s.scope == scope)
}

fn infer_scope_from_last_step(steps: &[InvestigationStep]) -> Option<Vec<(String, String)>> {
    let last = steps.last()?;
    let mover = last.movers.iter().find(|m| {
        !m.segment.trim().is_empty() && m.segment != "(blank)" && m.segment != "(unknown)"
    })?;
    let mut scope = last.scope.clone();
    scope.push((last.dimension.clone(), mover.segment.clone()));
    Some(scope)
}

fn infer_group_by_from_context(
    action: &str,
    steps: &[InvestigationStep],
    available_dimensions: &[String],
    scope: &[(String, String)],
) -> Option<String> {
    if available_dimensions.is_empty() {
        return None;
    }
    if steps.is_empty() {
        return available_dimensions.first().cloned();
    }

    let mut used = scope
        .iter()
        .map(|(k, _)| k.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    if let Some(last) = steps.last() {
        used.insert(last.dimension.to_ascii_lowercase());
    }

    if action == "drill_down" || !scope.is_empty() {
        if let Some(next) = available_dimensions
            .iter()
            .find(|d| !used.contains(&d.to_ascii_lowercase()))
        {
            return Some(next.clone());
        }
    }

    available_dimensions.first().cloned()
}

fn metric_matches_expected(requested: &str, expected: &str) -> bool {
    let requested_norm = normalize_metric_name(requested);
    if requested_norm.is_empty() {
        return false;
    }
    metric_aliases(expected).contains(&requested_norm)
}

fn scope_has_dimension(scope: &[(String, String)], dimension: &str) -> bool {
    scope.iter().any(|(k, _)| k.eq_ignore_ascii_case(dimension))
}

fn scope_value<'a>(scope: &'a [(String, String)], dimension: &str) -> Option<&'a str> {
    scope
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(dimension))
        .map(|(_, v)| v.as_str())
}

fn conflicting_scope_binding(
    prior_scope: &[(String, String)],
    next_scope: &[(String, String)],
) -> Option<(String, String, String)> {
    for (dim, prev) in prior_scope {
        if let Some(next) = scope_value(next_scope, dim) {
            if next != prev {
                return Some((dim.clone(), prev.clone(), next.to_string()));
            }
        }
    }
    None
}

fn scope_is_superset_of(
    candidate_scope: &[(String, String)],
    required_scope: &[(String, String)],
) -> bool {
    required_scope.iter().all(|(dim, value)| {
        scope_value(candidate_scope, dim)
            .map(|v| v == value)
            .unwrap_or(false)
    })
}

fn merge_scope(
    required_scope: &[(String, String)],
    candidate_scope: &[(String, String)],
) -> Vec<(String, String)> {
    let mut merged = required_scope.to_vec();
    for (dim, value) in candidate_scope {
        if !scope_has_dimension(&merged, dim) {
            merged.push((dim.clone(), value.clone()));
        }
    }
    merged
}

fn normalize_metric_name(s: &str) -> String {
    s.to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
}

fn metric_aliases(expected: &str) -> Vec<String> {
    let mut out = Vec::<String>::new();
    let expected_norm = normalize_metric_name(expected);
    if !expected_norm.is_empty() {
        out.push(expected_norm);
    }

    let lower = expected.to_ascii_lowercase();
    if let Some(base) = lower.strip_suffix("_usd") {
        let a = normalize_metric_name(base);
        if !a.is_empty() {
            out.push(a);
        }
    }
    if let Some(base) = lower.strip_suffix("_amount") {
        let a = normalize_metric_name(base);
        if !a.is_empty() {
            out.push(a);
        }
    }
    if let Some(base) = lower.strip_suffix("_pct") {
        let a = normalize_metric_name(base);
        if !a.is_empty() {
            out.push(a);
        }
    }
    if let Some(base) = lower.strip_suffix("_rate") {
        let a = normalize_metric_name(base);
        if !a.is_empty() {
            out.push(a);
        }
    }
    if let Some(base) = lower.strip_suffix("_ratio") {
        let a = normalize_metric_name(base);
        if !a.is_empty() {
            out.push(a);
        }
    }

    if let Some((first, _)) = lower.split_once('_') {
        let a = normalize_metric_name(first);
        if !a.is_empty() {
            out.push(a);
        }
    }

    out.sort();
    out.dedup();
    out
}

fn deterministic_fallback_action(
    args: &InvestigateWorkflowArgs,
    mode: InvestigationMode,
    input_kind: InvestigateInputKind,
    top_dimension: &str,
    drill_dimensions: &[String],
    steps: &[InvestigationStep],
) -> InvestigationExecAction {
    if steps.is_empty() {
        return InvestigationExecAction::AnalyzeCompare {
            group_by: top_dimension.to_string(),
            scope: vec![],
            reason: "fallback to top-level compare".to_string(),
        };
    }
    if input_kind == InvestigateInputKind::JsonArtifacts {
        return InvestigationExecAction::Stop {
            reason: "fallback stop: JSON artifact mode cannot drill down".to_string(),
        };
    }
    let last = match steps.last() {
        Some(s) => s,
        None => {
            return InvestigationExecAction::Stop {
                reason: "fallback stop: missing previous step".to_string(),
            };
        }
    };
    let used_dims = used_dimensions_for_step(last);
    let next_dimension = match choose_next_dimension(drill_dimensions, &used_dims) {
        Some(d) => d,
        None => {
            return InvestigationExecAction::Stop {
                reason: "fallback stop: no remaining drill dimension".to_string(),
            };
        }
    };
    let candidate = select_drill_candidates(
        last,
        mode,
        args.min_contribution,
        args.min_delta_abs,
        args.min_slice_rows,
        1,
    )
    .into_iter()
    .find(|m| !m.segment.contains(" | ") && m.segment != "(blank)");
    let Some(mover) = candidate else {
        return InvestigationExecAction::Stop {
            reason: "fallback stop: no eligible drill candidate".to_string(),
        };
    };

    let mut scope = last.scope.clone();
    scope.push((last.dimension.clone(), mover.segment.clone()));
    InvestigationExecAction::DrillDown {
        group_by: next_dimension,
        scope,
        reason: format!("fallback drill on {}='{}'", last.dimension, mover.segment),
    }
}

fn investigate_available_dimensions(
    top_dimension: &str,
    drill_dimensions: &[String],
) -> Vec<String> {
    let mut out = Vec::<String>::new();
    out.push(top_dimension.to_string());
    for d in drill_dimensions {
        if !out.iter().any(|x| x.eq_ignore_ascii_case(d)) {
            out.push(d.clone());
        }
    }
    out
}

fn major_change_limit_for_dimensions(available_dimensions: &[String]) -> usize {
    available_dimensions.len().clamp(3, 10)
}

fn identify_major_global_changes(
    args: &InvestigateWorkflowArgs,
    input: &InvestigationInputRefs<'_>,
    mode: InvestigationMode,
    available_dimensions: &[String],
    limit: usize,
) -> Result<Vec<InvestigationMajorChange>> {
    let mut changes = Vec::<InvestigationMajorChange>::new();
    let mut seen = HashSet::<String>::new();
    let root_scope = Vec::<(String, String)>::new();

    for dimension in available_dimensions {
        let key = dimension.to_ascii_lowercase();
        if !seen.insert(key) {
            continue;
        }
        let step = investigation_step_from_inputs(args, input, dimension, &root_scope, mode)?;
        let Some(top) = step.movers.iter().find(|m| {
            !m.segment.trim().is_empty() && m.segment != "(blank)" && m.segment != "(unknown)"
        }) else {
            continue;
        };
        let score = mover_score(top, mode);
        changes.push(InvestigationMajorChange {
            dimension: step.dimension.clone(),
            segment: top.segment.clone(),
            primary_metric: step.primary_metric.clone(),
            delta_primary_metric_value: top.delta_primary_metric_value,
            delta_share_pp: top.delta_share_pp,
            score,
        });
    }

    changes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.dimension.cmp(&b.dimension))
            .then_with(|| a.segment.cmp(&b.segment))
    });
    changes.truncate(limit.max(1));
    Ok(changes)
}

fn resolve_dimension_name(requested: &str, available_dimensions: &[String]) -> Option<String> {
    available_dimensions
        .iter()
        .find(|d| d.eq_ignore_ascii_case(requested))
        .cloned()
}

fn ordered_scope_from_filters(
    filters: Option<&HashMap<String, String>>,
    available_dimensions: &[String],
) -> Result<Vec<(String, String)>> {
    let Some(filters) = filters else {
        return Ok(vec![]);
    };
    for key in filters.keys() {
        if resolve_dimension_name(key, available_dimensions).is_none() {
            return Err(anyhow!("llm filter uses unknown dimension '{}'", key));
        }
    }
    let mut scope = Vec::<(String, String)>::new();
    for dim in available_dimensions {
        let value = filters
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(dim))
            .map(|(_, v)| v.as_str());
        if let Some(v) = value {
            if v.trim().is_empty() {
                return Err(anyhow!("llm filter for '{}' is blank", dim));
            }
            scope.push((dim.clone(), v.trim().to_string()));
        }
    }
    Ok(scope)
}

fn format_scope(scope: &[(String, String)]) -> String {
    if scope.is_empty() {
        "global".to_string()
    } else {
        scope
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn compact_error_message(msg: &str) -> String {
    let lines = msg
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>();

    let preferred = lines
        .iter()
        .copied()
        .find(|l| {
            let lower = l.to_ascii_lowercase();
            lower.contains("error")
                || lower.contains("failed")
                || lower.contains("invalid")
                || lower.contains("unknown")
        })
        .or_else(|| {
            lines.iter().copied().find(|l| {
                !(l.starts_with("load_backend:")
                    || l.starts_with("print_info:")
                    || l.starts_with("llama_model_loader:")
                    || l.starts_with("llama_context:")
                    || l.starts_with("sched_reserve:")
                    || l.starts_with("system_info:")
                    || l.starts_with("sampler ")
                    || l.starts_with("generate:"))
            })
        })
        .or_else(|| lines.last().copied())
        .unwrap_or("unknown error");

    let mut out = preferred.to_string();
    let max_len = 180usize;
    if out.len() > max_len {
        out.truncate(max_len.saturating_sub(3));
        out.push_str("...");
    }
    out
}

fn compact_trace_preview(text: &str, max_len: usize) -> String {
    let mut out = text.replace('\r', " ").replace('\n', "\\n");
    if out.len() > max_len {
        out.truncate(max_len.saturating_sub(3));
        out.push_str("...");
    }
    out
}

fn resolve_planner_model(args: &InvestigateWorkflowArgs) -> Result<String> {
    if let Some(model) = args.planner_model.clone() {
        if !model.trim().is_empty() {
            return Ok(model);
        }
    }
    match args.planner_backend {
        BackendArg::Bedrock => Ok("anthropic.claude-3-haiku-20240307-v1:0".to_string()),
        BackendArg::Local => local_planner_model_from_env().ok_or_else(|| {
            anyhow!(
                "--planner-model is required when --planner llm --planner-backend local (or set FACTORLENS_PLANNER_LOCAL_MODEL / FACTORLENS_LOCAL_MODEL)"
            )
        }),
    }
}

fn local_planner_model_from_env() -> Option<String> {
    for key in ["FACTORLENS_PLANNER_LOCAL_MODEL", "FACTORLENS_LOCAL_MODEL"] {
        if let Ok(raw) = std::env::var(key) {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn local_fallback_planner_model_name(args: &InvestigateWorkflowArgs) -> Option<String> {
    if args.planner_backend == BackendArg::Bedrock {
        return local_planner_model_from_env();
    }
    None
}

fn build_local_fallback_planner(args: &InvestigateWorkflowArgs) -> Option<Box<dyn LlmClient>> {
    if args.planner_backend != BackendArg::Bedrock {
        return None;
    }
    let model = local_planner_model_from_env()?;
    Some(build_client(Backend::Local, model))
}

fn planner_backend_label(backend: BackendArg) -> &'static str {
    match backend {
        BackendArg::Local => "local",
        BackendArg::Bedrock => "bedrock",
    }
}

fn llm_finalize_summary(
    planner: &dyn LlmClient,
    question: &str,
    steps: &[InvestigationStep],
    trace: &[InvestigationTraceStep],
    stop_reason: &str,
) -> Result<String> {
    let summary_input = serde_json::json!({
        "question": question,
        "steps": steps,
        "trace": trace,
        "stop_reason": stop_reason
    });
    let system_prompt =
        "Summarize investigation findings. Use only provided facts. Do not invent numbers. Use at most 2 decimal places. Do not output commands or code.";
    let user_prompt = format!(
        "Provide 2-4 short lines. Keep it concise and grounded. Do not output JSON.\n{}",
        serde_json::to_string_pretty(&summary_input)?
    );
    let raw = planner.answer(system_prompt, &user_prompt)?;
    let cleaned = sanitize_llm_summary(&raw, question);
    if cleaned.is_empty() {
        return Err(anyhow!("llm final summary was empty"));
    }
    if !llm_summary_is_usable(&cleaned, steps, stop_reason) {
        return Err(anyhow!("llm final summary failed quality checks"));
    }
    Ok(cleaned)
}

fn sanitize_llm_summary(raw: &str, question: &str) -> String {
    let mut text = raw.trim();
    if let Some((_, tail)) = raw.rsplit_once("assistant") {
        let trimmed = tail.trim();
        if !trimmed.is_empty() {
            text = trimmed;
        }
    }

    let mut out = Vec::<String>::new();
    let mut seen = HashSet::<String>::new();
    let mut seen_prefix = HashSet::<String>::new();
    for line in text.lines() {
        let trimmed = line.trim().trim_start_matches('-').trim();
        let trimmed = trimmed.trim_end_matches('|').trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case(question.trim()) {
            continue;
        }

        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("provide 2-4 short lines")
            || lower.starts_with("here is a concise summary")
            || lower.starts_with("summarize investigation findings")
            || lower.starts_with("note:")
            || lower.starts_with("the final answer is")
            || lower == "assistant"
            || lower.starts_with("input:")
        {
            continue;
        }
        if lower.starts_with("|")
            || lower.starts_with("$ ")
            || lower.starts_with("jq ")
            || lower.contains("| jq")
            || lower.contains("cargo run")
            || lower.contains("python3 ")
        {
            continue;
        }
        if lower.starts_with("\"question\"")
            || lower.starts_with("\"steps\"")
            || lower.starts_with("\"trace\"")
            || lower.starts_with("\"stop_reason\"")
            || lower.starts_with("\"action\"")
            || lower.starts_with("\"decision\"")
            || lower.starts_with("\"depth\"")
        {
            continue;
        }
        if trimmed.starts_with('{')
            || trimmed.starts_with('}')
            || trimmed.starts_with('[')
            || trimmed.starts_with(']')
            || trimmed.starts_with('"')
        {
            continue;
        }
        let alpha_count = trimmed.chars().filter(|c| c.is_ascii_alphabetic()).count();
        if alpha_count < 8 {
            continue;
        }
        let punctuation = trimmed
            .chars()
            .filter(|c| matches!(c, '{' | '}' | '[' | ']' | '"' | ':' | ','))
            .count();
        if punctuation * 2 >= trimmed.len() {
            continue;
        }
        let canonical = trimmed.to_ascii_lowercase();
        let canonical_ws = canonical.split_whitespace().collect::<Vec<_>>().join(" ");
        let prefix = canonical_ws.chars().take(64).collect::<String>();
        if seen.insert(canonical) && seen_prefix.insert(prefix) {
            out.push(trimmed.to_string());
        }
        if out.len() >= 4 {
            break;
        }
    }
    out.join("\n")
}

fn llm_summary_is_usable(summary: &str, steps: &[InvestigationStep], stop_reason: &str) -> bool {
    if summary.trim().is_empty() {
        return false;
    }
    if summary_looks_truncated(summary) {
        return false;
    }
    let lower = summary.to_ascii_lowercase();
    if lower.contains("[user]")
        || lower.contains("[system]")
        || lower.contains("```")
        || lower.contains("| jq")
        || lower.contains("the final answer is")
    {
        return false;
    }
    if contains_long_decimal(summary, 2) {
        return false;
    }

    summary_numbers_are_grounded(
        summary,
        &collect_allowed_summary_numbers(steps, stop_reason),
    )
}

fn summary_looks_truncated(summary: &str) -> bool {
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        return false;
    }
    let Some(last_line) = trimmed.lines().rev().find(|l| !l.trim().is_empty()) else {
        return false;
    };
    let tail = last_line.trim().trim_end_matches('|').trim_end();
    if tail.is_empty() {
        return true;
    }
    let Some(last_char) = tail.chars().last() else {
        return true;
    };
    if matches!(last_char, '.' | '!' | '?' | ')' | ']' | '"' | '\'') {
        return false;
    }
    if !last_char.is_ascii_alphanumeric() {
        return false;
    }
    let alpha_count = tail.chars().filter(|c| c.is_ascii_alphabetic()).count();
    alpha_count >= 10
}

fn contains_long_decimal(text: &str, max_decimals: usize) -> bool {
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            let mut dec = 0usize;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                dec += 1;
                i += 1;
            }
            if dec > max_decimals {
                return true;
            }
        }
    }
    false
}

fn collect_allowed_summary_numbers(steps: &[InvestigationStep], stop_reason: &str) -> Vec<f64> {
    let mut out = Vec::<f64>::new();
    for step in steps {
        out.push(step.depth as f64);
        out.push(step.base_records as f64);
        out.push(step.new_records as f64);
        out.push(step.segment_count as f64);
        out.push(step.top5_concentration_base_pct);
        out.push(step.top5_concentration_new_pct);
        out.push(step.top5_concentration_delta_pp);
        out.push(step.top1_concentration_base_pct);
        out.push(step.top1_concentration_new_pct);
        out.push(step.top1_concentration_delta_pp);
        for mover in &step.movers {
            out.push(mover.base_records as f64);
            out.push(mover.new_records as f64);
            out.push(mover.base_share_pct);
            out.push(mover.new_share_pct);
            out.push(mover.delta_share_pp);
            out.push(mover.base_primary_metric_value);
            out.push(mover.new_primary_metric_value);
            out.push(mover.delta_primary_metric_value);
        }
    }
    out.extend(extract_numeric_values(stop_reason));
    out.push(1.0);
    out.push(5.0);
    out
}

fn summary_numbers_are_grounded(summary: &str, allowed: &[f64]) -> bool {
    if allowed.is_empty() {
        return true;
    }
    extract_numeric_values(summary)
        .into_iter()
        .all(|v| allowed.iter().any(|a| approx_number_match(v, *a)))
}

fn approx_number_match(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs());
    let tol = if scale >= 1_000_000.0 {
        1.0
    } else if scale >= 1_000.0 {
        0.5
    } else {
        0.05
    };
    (a - b).abs() <= tol
}

fn extract_numeric_values(text: &str) -> Vec<f64> {
    let mut out = Vec::<f64>::new();
    let chars = text.chars().collect::<Vec<_>>();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if !(c.is_ascii_digit() || c == '-') {
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        while i < chars.len() {
            let ch = chars[i];
            if ch.is_ascii_digit() || ch == '.' || ch == ',' || ch == '_' {
                i += 1;
            } else {
                break;
            }
        }
        let token = chars[start..i].iter().collect::<String>();
        let normalized = token.replace([',', '_'], "");
        if normalized == "-" || normalized == "." || normalized == "-." {
            continue;
        }
        if let Ok(v) = normalized.parse::<f64>() {
            out.push(v);
        }
    }
    out
}

fn llm_finalize_summary_with_fallback(
    planner: &dyn LlmClient,
    local_fallback: Option<&dyn LlmClient>,
    question: &str,
    steps: &[InvestigationStep],
    trace: &[InvestigationTraceStep],
    stop_reason: &str,
) -> Result<String> {
    match llm_finalize_summary(planner, question, steps, trace, stop_reason) {
        Ok(s) => Ok(s),
        Err(primary_err) => {
            if let Some(local) = local_fallback {
                llm_finalize_summary(local, question, steps, trace, stop_reason).map_err(
                    |local_err| {
                        anyhow!(
                            "primary summary planner failed: {}; local fallback failed: {}",
                            primary_err,
                            local_err
                        )
                    },
                )
            } else {
                Err(primary_err)
            }
        }
    }
}

fn deterministic_summary_from_steps(
    steps: &[InvestigationStep],
    stop_reason: &str,
    recommended_next_question: &str,
) -> String {
    if let Some(step0) = steps.first() {
        if let Some(top) = step0.movers.first() {
            return format!(
                "Largest shift is '{}' ({:+.2} delta metric, {:+.2} pp share).\nStop reason: {}.\nNext: {}",
                top.segment,
                top.delta_primary_metric_value,
                top.delta_share_pp,
                stop_reason,
                recommended_next_question
            );
        }
    }
    format!(
        "No significant mover was identified.\nStop reason: {}.\nNext: {}",
        stop_reason, recommended_next_question
    )
}

fn resolve_investigation_dimensions(
    args: &InvestigateWorkflowArgs,
    base_path: &Path,
    new_path: &Path,
    input_kind: InvestigateInputKind,
    base_artifact: Option<&serde_json::Value>,
    new_artifact: Option<&serde_json::Value>,
) -> Result<(String, Vec<String>)> {
    if input_kind == InvestigateInputKind::JsonArtifacts {
        let inferred = infer_artifact_group_dimension(base_artifact, new_artifact)?;
        if let Some(requested) = args.dimensions.first() {
            if !requested.eq_ignore_ascii_case(&inferred) {
                return Err(anyhow!(
                    "for JSON artifacts, --dimensions must match artifact grouping '{}' (got '{}')",
                    inferred,
                    requested
                ));
            }
        }
        let drill_dimensions = if args.drill_fields.is_empty() {
            args.dimensions.iter().skip(1).cloned().collect::<Vec<_>>()
        } else {
            args.drill_fields.clone()
        };
        return Ok((inferred, drill_dimensions));
    }

    if !args.dimensions.is_empty() {
        let top_dimension = args
            .dimensions
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("expected at least one dimension"))?;
        let drill_dimensions = if args.drill_fields.is_empty() {
            args.dimensions.iter().skip(1).cloned().collect::<Vec<_>>()
        } else {
            args.drill_fields.clone()
        };
        return Ok((top_dimension, drill_dimensions));
    }

    let inferred = infer_csv_investigation_dimensions(base_path, new_path, args.metric.as_deref())?;
    let top_dimension = inferred
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("failed to infer top dimension for investigate"))?;
    let drill_dimensions = if args.drill_fields.is_empty() {
        inferred.iter().skip(1).cloned().collect::<Vec<_>>()
    } else {
        args.drill_fields.clone()
    };
    Ok((top_dimension, drill_dimensions))
}

fn infer_csv_investigation_dimensions(
    base_path: &Path,
    new_path: &Path,
    metric: Option<&str>,
) -> Result<Vec<String>> {
    const AUTO_KEEP_HIGH_CARD_TOP1_COUNT_SHARE: f64 = 0.12;
    const AUTO_KEEP_HIGH_CARD_TOP5_COUNT_SHARE: f64 = 0.35;
    const AUTO_KEEP_HIGH_CARD_TOP1_METRIC_SHARE: f64 = 0.12;
    const AUTO_KEEP_HIGH_CARD_TOP5_METRIC_SHARE: f64 = 0.40;

    let sample_rows = 300usize;
    let (base_headers, base_rows) = read_csv_headers_and_sample(base_path, sample_rows)?;
    let (new_headers, new_rows) = read_csv_headers_and_sample(new_path, sample_rows)?;
    let base_metric_idx = metric.and_then(|m| header_index_case_insensitive(&base_headers, m));
    let new_metric_idx = metric.and_then(|m| header_index_case_insensitive(&new_headers, m));

    let mut out = Vec::<String>::new();
    for base_name in &base_headers {
        if metric.is_some_and(|m| base_name.eq_ignore_ascii_case(m)) {
            continue;
        }
        if looks_like_identifier_column(base_name) {
            continue;
        }
        let Some(base_idx) = header_index_case_insensitive(&base_headers, base_name) else {
            continue;
        };
        let Some(new_idx) = header_index_case_insensitive(&new_headers, base_name) else {
            continue;
        };

        let mut non_empty = 0usize;
        let mut numeric_like = 0usize;
        let mut date_like = 0usize;
        let mut unique_values = HashSet::<String>::new();
        let mut value_counts = HashMap::<String, usize>::new();
        let mut metric_by_value = HashMap::<String, f64>::new();
        let mut metric_total_abs = 0.0f64;
        for rec in &base_rows {
            let v = rec.get(base_idx).unwrap_or("").trim();
            if v.is_empty() {
                continue;
            }
            non_empty += 1;
            *value_counts.entry(v.to_string()).or_insert(0) += 1;
            if parse_numeric(v).is_some() {
                numeric_like += 1;
            }
            if parse_date_like(v).is_some() {
                date_like += 1;
            }
            if unique_values.len() < 256 {
                unique_values.insert(v.to_string());
            }
            if let Some(metric_idx) = base_metric_idx {
                if let Some(m) = parse_numeric(rec.get(metric_idx).unwrap_or("").trim()) {
                    let abs = m.abs();
                    metric_total_abs += abs;
                    *metric_by_value.entry(v.to_string()).or_insert(0.0) += abs;
                }
            }
        }
        for rec in &new_rows {
            let v = rec.get(new_idx).unwrap_or("").trim();
            if v.is_empty() {
                continue;
            }
            non_empty += 1;
            *value_counts.entry(v.to_string()).or_insert(0) += 1;
            if parse_numeric(v).is_some() {
                numeric_like += 1;
            }
            if parse_date_like(v).is_some() {
                date_like += 1;
            }
            if unique_values.len() < 256 {
                unique_values.insert(v.to_string());
            }
            if let Some(metric_idx) = new_metric_idx {
                if let Some(m) = parse_numeric(rec.get(metric_idx).unwrap_or("").trim()) {
                    let abs = m.abs();
                    metric_total_abs += abs;
                    *metric_by_value.entry(v.to_string()).or_insert(0.0) += abs;
                }
            }
        }
        if non_empty == 0 {
            continue;
        }

        let numeric_ratio = numeric_like as f64 / non_empty as f64;
        let date_ratio = date_like as f64 / non_empty as f64;
        let unique_count = unique_values.len();
        let unique_ratio = unique_count as f64 / non_empty as f64;
        let low_cardinality_code = unique_count > 0
            && unique_count <= 20
            && (unique_count as f64 / non_empty as f64) <= 0.2;
        // High-cardinality text-like columns are usually IDs/keys and make poor drill dimensions.
        let high_cardinality_text =
            numeric_ratio < 0.9 && unique_count >= 64 && unique_ratio >= 0.35;
        let mut count_distribution = value_counts.values().copied().collect::<Vec<_>>();
        count_distribution.sort_unstable_by(|a, b| b.cmp(a));
        let top1_count_share = count_distribution
            .first()
            .map(|v| *v as f64 / non_empty as f64)
            .unwrap_or(0.0);
        let top5_count_share =
            count_distribution.iter().take(5).sum::<usize>() as f64 / non_empty as f64;

        let mut metric_distribution = metric_by_value.values().copied().collect::<Vec<_>>();
        metric_distribution.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top1_metric_share = if metric_total_abs > 0.0 {
            metric_distribution.first().copied().unwrap_or(0.0) / metric_total_abs
        } else {
            0.0
        };
        let top5_metric_share = if metric_total_abs > 0.0 {
            metric_distribution.iter().take(5).sum::<f64>() / metric_total_abs
        } else {
            0.0
        };
        let keep_high_cardinality_text = top1_count_share >= AUTO_KEEP_HIGH_CARD_TOP1_COUNT_SHARE
            || top5_count_share >= AUTO_KEEP_HIGH_CARD_TOP5_COUNT_SHARE
            || top1_metric_share >= AUTO_KEEP_HIGH_CARD_TOP1_METRIC_SHARE
            || top5_metric_share >= AUTO_KEEP_HIGH_CARD_TOP5_METRIC_SHARE;

        if date_ratio >= 0.8
            || (numeric_ratio >= 0.9 && !low_cardinality_code)
            || (high_cardinality_text && !keep_high_cardinality_text)
        {
            continue;
        }
        out.push(base_name.to_string());
    }

    if out.is_empty() {
        return Err(anyhow!(
            "could not auto-detect categorical dimensions from CSVs; pass --dimensions explicitly"
        ));
    }
    Ok(out)
}

fn looks_like_identifier_column(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n == "id"
        || n.ends_with("_id")
        || n.ends_with("_identifier")
        || n.ends_with("_uuid")
        || n.ends_with("_guid")
        || n.ends_with("_key")
        || n.ends_with("_hash")
}

fn read_csv_headers_and_sample(
    path: &Path,
    sample_rows: usize,
) -> Result<(StringRecord, Vec<StringRecord>)> {
    let mut rdr = csv::Reader::from_path(path)
        .map_err(|e| anyhow!("failed to read csv '{}': {}", path.display(), e))?;
    let headers = rdr
        .headers()
        .map_err(|e| anyhow!("failed to read headers from '{}': {}", path.display(), e))?
        .clone();
    let mut rows = Vec::<StringRecord>::new();
    for rec in rdr.records().take(sample_rows) {
        rows.push(
            rec.map_err(|e| anyhow!("failed to read rows from '{}': {}", path.display(), e))?,
        );
    }
    Ok((headers, rows))
}

fn header_index_case_insensitive(headers: &StringRecord, name: &str) -> Option<usize> {
    headers.iter().position(|h| h.eq_ignore_ascii_case(name))
}

fn detect_investigate_input_kind(base: &Path, new: &Path) -> Result<InvestigateInputKind> {
    let base_json = base
        .extension()
        .map(|x| x.to_string_lossy().to_ascii_lowercase() == "json")
        .unwrap_or(false);
    let new_json = new
        .extension()
        .map(|x| x.to_string_lossy().to_ascii_lowercase() == "json")
        .unwrap_or(false);
    match (base_json, new_json) {
        (true, true) => Ok(InvestigateInputKind::JsonArtifacts),
        (false, false) => Ok(InvestigateInputKind::CsvDatasets),
        _ => Err(anyhow!(
            "--base/--new must both be JSON artifacts or both be CSV datasets"
        )),
    }
}

fn read_json_file(path: &Path, label: &str) -> Result<serde_json::Value> {
    let txt = fs::read_to_string(path)
        .map_err(|e| anyhow!("failed to read {} json '{}': {}", label, path.display(), e))?;
    serde_json::from_str(&txt)
        .map_err(|e| anyhow!("failed to parse {} json '{}': {}", label, path.display(), e))
}

fn infer_artifact_group_dimension(
    base_artifact: Option<&serde_json::Value>,
    new_artifact: Option<&serde_json::Value>,
) -> Result<String> {
    let base_dim = base_artifact
        .and_then(|v| v.get("group_by"))
        .and_then(|x| x.as_array())
        .and_then(|xs| xs.first())
        .and_then(|x| x.as_str())
        .map(|x| x.to_string());
    let new_dim = new_artifact
        .and_then(|v| v.get("group_by"))
        .and_then(|x| x.as_array())
        .and_then(|xs| xs.first())
        .and_then(|x| x.as_str())
        .map(|x| x.to_string());

    if let (Some(b), Some(n)) = (&base_dim, &new_dim) {
        if b != n {
            return Err(anyhow!(
                "base/new artifacts use different group_by dimensions ('{}' vs '{}')",
                b,
                n
            ));
        }
    }

    new_dim.or(base_dim).ok_or_else(|| {
        anyhow!(
            "could not infer grouping dimension from artifact JSON; pass --dimensions explicitly"
        )
    })
}

fn artifact_has_metric(v: &serde_json::Value, metric: &str) -> bool {
    let metric_in_declared = v
        .get("metrics")
        .and_then(|x| x.as_array())
        .map(|xs| {
            xs.iter().filter_map(|x| x.as_str()).any(|m| {
                m == metric
                    || m.eq_ignore_ascii_case(metric)
                    || m.strip_suffix("_p25")
                        .or_else(|| m.strip_suffix("_p50"))
                        .or_else(|| m.strip_suffix("_p75"))
                        .is_some_and(|base| base.eq_ignore_ascii_case(metric))
            })
        })
        .unwrap_or(false);
    if metric_in_declared {
        return true;
    }
    v.get("groups")
        .and_then(|x| x.as_array())
        .and_then(|xs| xs.first())
        .and_then(|g| value_by_key_case_insensitive(g, metric))
        .is_some()
}

fn validate_metric_in_artifact(v: &serde_json::Value, metric: &str, label: &str) -> Result<()> {
    if artifact_has_metric(v, metric) {
        return Ok(());
    }
    Err(anyhow!(
        "{} artifact does not contain metric '{}' in groups/metrics",
        label,
        metric
    ))
}

fn artifact_primary_metric(v: &serde_json::Value) -> Option<String> {
    v.get("primary_metric")
        .and_then(|x| x.as_str())
        .map(|x| x.to_string())
}

fn resolve_metric_for_json_artifacts(
    base: &serde_json::Value,
    new: &serde_json::Value,
    requested_metric: Option<&str>,
) -> Result<String> {
    if let Some(metric) = requested_metric {
        validate_metric_in_artifact(base, metric, "base")?;
        validate_metric_in_artifact(new, metric, "new")?;
        return Ok(metric.to_string());
    }

    let base_primary = artifact_primary_metric(base)
        .ok_or_else(|| anyhow!("base artifact missing primary_metric; pass --metric explicitly"))?;
    let new_primary = artifact_primary_metric(new)
        .ok_or_else(|| anyhow!("new artifact missing primary_metric; pass --metric explicitly"))?;
    if base_primary != new_primary {
        return Err(anyhow!(
            "base/new artifacts have different primary_metric values ('{}' vs '{}'); pass --metric that exists in both artifacts",
            base_primary,
            new_primary
        ));
    }
    validate_metric_in_artifact(base, &new_primary, "base")?;
    validate_metric_in_artifact(new, &new_primary, "new")?;
    Ok(new_primary)
}

struct InvestigationInputRefs<'a> {
    base_path: &'a Path,
    new_path: &'a Path,
    input_kind: InvestigateInputKind,
    base_artifact: Option<&'a serde_json::Value>,
    new_artifact: Option<&'a serde_json::Value>,
}

fn investigation_step_from_inputs(
    args: &InvestigateWorkflowArgs,
    input: &InvestigationInputRefs<'_>,
    dimension: &str,
    scope: &[(String, String)],
    mode: InvestigationMode,
) -> Result<InvestigationStep> {
    let top_n = args.top_movers.max(args.max_branches).max(1);
    let (base_report, new_report, metric_override) = match input.input_kind {
        InvestigateInputKind::JsonArtifacts => {
            let base = input
                .base_artifact
                .cloned()
                .ok_or_else(|| anyhow!("missing base artifact payload"))?;
            let new = input
                .new_artifact
                .cloned()
                .ok_or_else(|| anyhow!("missing new artifact payload"))?;
            let metric = resolve_metric_for_json_artifacts(&base, &new, args.metric.as_deref())?;
            (base, new, Some(metric))
        }
        InvestigateInputKind::CsvDatasets => {
            let metric = args.metric.clone().ok_or_else(|| {
                anyhow!("--metric is required when --base/--new point to CSV datasets")
            })?;
            let base_json = analyze_csv_for_investigate(
                input.base_path,
                dimension,
                &metric,
                scope,
                top_n.saturating_mul(4),
            )?;
            let new_json = analyze_csv_for_investigate(
                input.new_path,
                dimension,
                &metric,
                scope,
                top_n.saturating_mul(4),
            )?;
            (base_json, new_json, Some(metric))
        }
    };
    build_investigation_step(
        dimension,
        scope,
        mode,
        &base_report,
        &new_report,
        metric_override.as_deref(),
        top_n,
    )
}

fn analyze_csv_for_investigate(
    input: &Path,
    dimension: &str,
    metric: &str,
    scope: &[(String, String)],
    top_n: usize,
) -> Result<serde_json::Value> {
    let input_buf = input.to_path_buf();
    let where_clauses = scope
        .iter()
        .map(|(col, val)| format!("{}={}", col, val))
        .collect::<Vec<_>>();
    let report = analyze_table_csv(
        &input_buf,
        None,
        &[dimension.to_string()],
        3,
        &[metric.to_string()],
        false,
        AggKind::Sum,
        &[],
        false,
        false,
        None,
        &where_clauses,
        false,
        Some(metric),
        top_n.max(10),
        0,
        2,
        1,
        None,
        None,
        &[],
    )?;
    Ok(report.json)
}

fn build_investigation_step(
    dimension: &str,
    scope: &[(String, String)],
    mode: InvestigationMode,
    base: &serde_json::Value,
    new: &serde_json::Value,
    metric_hint: Option<&str>,
    top_movers: usize,
) -> Result<InvestigationStep> {
    let base_records = base.get("records").and_then(|x| x.as_u64()).unwrap_or(0);
    let new_records = new.get("records").and_then(|x| x.as_u64()).unwrap_or(0);
    let base_top5_count = base.get("top5_count").and_then(|x| x.as_u64()).unwrap_or(0);
    let new_top5_count = new.get("top5_count").and_then(|x| x.as_u64()).unwrap_or(0);
    let base_top5_pct = pct(base_top5_count, base_records);
    let new_top5_pct = pct(new_top5_count, new_records);

    let primary_metric = metric_hint
        .map(|m| m.to_string())
        .or_else(|| {
            new.get("primary_metric")
                .and_then(|x| x.as_str())
                .map(|x| x.to_string())
        })
        .or_else(|| {
            base.get("primary_metric")
                .and_then(|x| x.as_str())
                .map(|x| x.to_string())
        })
        .ok_or_else(|| anyhow!("primary metric missing in reports; pass --metric explicitly"))?;

    let base_map = groups_to_map(base, &primary_metric);
    let new_map = groups_to_map(new, &primary_metric);
    let mut keys = base_map.keys().cloned().collect::<HashSet<_>>();
    keys.extend(new_map.keys().cloned());
    let segment_count = keys.len();
    let base_top1_pct = base_map
        .values()
        .map(|(_, share, _)| *share)
        .fold(0.0, f64::max);
    let new_top1_pct = new_map
        .values()
        .map(|(_, share, _)| *share)
        .fold(0.0, f64::max);

    let mut movers = keys
        .into_iter()
        .map(|k| {
            let (bc, bs, bp) = base_map.get(&k).copied().unwrap_or((0, 0.0, 0.0));
            let (nc, ns, np) = new_map.get(&k).copied().unwrap_or((0, 0.0, 0.0));
            InvestigationMover {
                segment: k,
                base_records: bc,
                new_records: nc,
                base_share_pct: bs,
                new_share_pct: ns,
                delta_share_pp: ns - bs,
                base_primary_metric_value: bp,
                new_primary_metric_value: np,
                delta_primary_metric_value: np - bp,
            }
        })
        .collect::<Vec<_>>();
    movers.sort_by(|a, b| {
        mover_score(b, mode)
            .partial_cmp(&mover_score(a, mode))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.segment.cmp(&b.segment))
    });
    movers.truncate(top_movers.max(1));

    Ok(InvestigationStep {
        depth: scope.len(),
        dimension: dimension.to_string(),
        scope: scope.to_vec(),
        primary_metric,
        base_records,
        new_records,
        segment_count,
        top5_concentration_base_pct: base_top5_pct,
        top5_concentration_new_pct: new_top5_pct,
        top5_concentration_delta_pp: new_top5_pct - base_top5_pct,
        top1_concentration_base_pct: base_top1_pct,
        top1_concentration_new_pct: new_top1_pct,
        top1_concentration_delta_pp: new_top1_pct - base_top1_pct,
        movers,
    })
}

fn select_drill_candidates(
    step: &InvestigationStep,
    mode: InvestigationMode,
    min_contribution: f64,
    min_delta_abs: f64,
    min_slice_rows: u64,
    branch_limit: usize,
) -> Vec<&InvestigationMover> {
    let mut candidates = step
        .movers
        .iter()
        .filter(|m| {
            drill_candidate_matches_base_thresholds(m, mode, min_contribution, min_slice_rows)
                && (mode == InvestigationMode::ConcentrationDrivers
                    || m.delta_primary_metric_value.abs() + f64::EPSILON >= min_delta_abs)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        mover_score(b, mode)
            .partial_cmp(&mover_score(a, mode))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.segment.cmp(&b.segment))
    });
    candidates.truncate(branch_limit.max(1));
    candidates
}

fn used_dimensions_for_step(step: &InvestigationStep) -> Vec<String> {
    let mut used = Vec::<String>::new();
    for (dim, _) in &step.scope {
        if !used.iter().any(|x| x == dim) {
            used.push(dim.clone());
        }
    }
    if !used.iter().any(|x| x == &step.dimension) {
        used.push(step.dimension.clone());
    }
    used
}

fn mover_score(mover: &InvestigationMover, mode: InvestigationMode) -> f64 {
    match mode {
        InvestigationMode::ConcentrationDrivers => mover.delta_share_pp.abs(),
        InvestigationMode::ChangeDrivers
        | InvestigationMode::CompareSnapshots
        | InvestigationMode::RecommendNext => mover.delta_primary_metric_value.abs(),
    }
}

fn drill_candidate_matches_base_thresholds(
    mover: &InvestigationMover,
    mode: InvestigationMode,
    min_contribution: f64,
    min_slice_rows: u64,
) -> bool {
    !mover.segment.trim().is_empty()
        && mover.segment != "(blank)"
        && mover.segment != "(unknown)"
        && mover.base_records.max(mover.new_records) >= min_slice_rows
        && mover_score(mover, mode) >= min_contribution
}

fn top_step_score(step: &InvestigationStep, mode: InvestigationMode) -> f64 {
    step.movers
        .first()
        .map(|m| mover_score(m, mode))
        .unwrap_or(0.0)
}

fn top_step_delta_abs(step: &InvestigationStep) -> f64 {
    step.movers
        .first()
        .map(|m| m.delta_primary_metric_value.abs())
        .unwrap_or(0.0)
}

fn step_total_delta(step: &InvestigationStep) -> f64 {
    let base_total = step
        .movers
        .iter()
        .map(|m| m.base_primary_metric_value)
        .sum::<f64>();
    let new_total = step
        .movers
        .iter()
        .map(|m| m.new_primary_metric_value)
        .sum::<f64>();
    new_total - base_total
}

fn build_investigation_coverage(steps: &[InvestigationStep]) -> InvestigationCoverage {
    let top_level_total_delta = steps.first().map(step_total_delta).unwrap_or(0.0);
    let total_delta_abs = top_level_total_delta.abs();
    let top_mover = steps
        .first()
        .and_then(|s| s.movers.first())
        .map(|m| (m.segment.clone(), m.delta_primary_metric_value.abs()))
        .unwrap_or_else(|| ("".to_string(), 0.0));
    let top_level_strongest_explained_pct = if total_delta_abs <= f64::EPSILON {
        0.0
    } else {
        (top_mover.1 / total_delta_abs) * 100.0
    };

    let mut step_coverage = Vec::<InvestigationCoverageStep>::new();
    for (idx, step) in steps.iter().enumerate() {
        let Some(top) = step.movers.first() else {
            continue;
        };
        let strongest_abs = top.delta_primary_metric_value.abs();
        let strongest_explained_pct_of_total_delta = if total_delta_abs <= f64::EPSILON {
            0.0
        } else {
            (strongest_abs / total_delta_abs) * 100.0
        };
        let residual_delta_abs_after_step = if total_delta_abs <= strongest_abs {
            0.0
        } else {
            total_delta_abs - strongest_abs
        };
        step_coverage.push(InvestigationCoverageStep {
            step_index: idx,
            depth: step.depth,
            scope: step.scope.clone(),
            dimension: step.dimension.clone(),
            strongest_segment: top.segment.clone(),
            strongest_delta_primary_metric_value: top.delta_primary_metric_value,
            strongest_delta_share_pp: top.delta_share_pp,
            strongest_explained_pct_of_total_delta,
            residual_delta_abs_after_step,
        });
    }

    InvestigationCoverage {
        total_delta_abs,
        top_level_total_delta,
        top_level_strongest_segment: if top_mover.0.is_empty() {
            None
        } else {
            Some(top_mover.0)
        },
        top_level_strongest_delta_abs: top_mover.1,
        top_level_strongest_explained_pct,
        step_coverage,
    }
}

fn scope_starts_with(scope: &[(String, String)], prefix: &[(String, String)]) -> bool {
    if prefix.len() > scope.len() {
        return false;
    }
    scope
        .iter()
        .zip(prefix.iter())
        .all(|(a, b)| a.0 == b.0 && a.1 == b.1)
}

fn investigation_scope_key(scope: &[(String, String)]) -> String {
    if scope.is_empty() {
        "global".to_string()
    } else {
        scope
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn build_investigation_branch_graph(
    steps: &[InvestigationStep],
    mode: InvestigationMode,
) -> InvestigationBranchGraph {
    let mut nodes = Vec::<InvestigationBranchNode>::new();
    for (idx, step) in steps.iter().enumerate() {
        let top = step.movers.first();
        let strongest_segment = top.map(|m| m.segment.clone());
        let strongest_delta_primary_metric_value =
            top.map(|m| m.delta_primary_metric_value).unwrap_or(0.0);
        let strongest_delta_share_pp = top.map(|m| m.delta_share_pp).unwrap_or(0.0);
        let score = top.map(|m| mover_score(m, mode)).unwrap_or(0.0);
        let id = format!(
            "d{}:{}:{}",
            step.depth,
            step.dimension,
            investigation_scope_key(&step.scope)
        );
        nodes.push(InvestigationBranchNode {
            id,
            step_index: idx,
            depth: step.depth,
            scope: step.scope.clone(),
            dimension: step.dimension.clone(),
            primary_metric: step.primary_metric.clone(),
            strongest_segment,
            strongest_delta_primary_metric_value,
            strongest_delta_share_pp,
            score,
        });
    }

    let mut edges = Vec::<InvestigationBranchEdge>::new();
    for child in &nodes {
        if child.depth == 0 || child.scope.is_empty() {
            continue;
        }
        let parent = nodes.iter().find(|candidate| {
            candidate.depth + 1 == child.depth
                && candidate.scope.len() + 1 == child.scope.len()
                && scope_starts_with(&child.scope, &candidate.scope)
                && child.scope[candidate.scope.len()].0 == candidate.dimension
        });
        if let Some(parent_node) = parent {
            edges.push(InvestigationBranchEdge {
                parent_id: parent_node.id.clone(),
                child_id: child.id.clone(),
                score_improvement: child.score - parent_node.score,
            });
        }
    }

    InvestigationBranchGraph { nodes, edges }
}

fn render_branch_graph_mermaid(branch_graph: &InvestigationBranchGraph) -> String {
    let mut out = String::new();
    out.push_str("```mermaid\n");
    out.push_str("graph TD\n");
    for node in &branch_graph.nodes {
        let scope_txt = if node.scope.is_empty() {
            "global".to_string()
        } else {
            node.scope
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let strongest = node
            .strongest_segment
            .as_deref()
            .map(str::to_string)
            .unwrap_or_else(|| "(none)".to_string());
        let label = format!(
            "d{} | {} | scope: {} | strongest: {}",
            node.depth, node.dimension, scope_txt, strongest
        )
        .replace('"', "'");
        out.push_str(&format!("  N{}[\"{}\"]\n", node.step_index, label));
    }
    for edge in &branch_graph.edges {
        let parent_idx = branch_graph
            .nodes
            .iter()
            .find(|n| n.id == edge.parent_id)
            .map(|n| n.step_index);
        let child_idx = branch_graph
            .nodes
            .iter()
            .find(|n| n.id == edge.child_id)
            .map(|n| n.step_index);
        if let (Some(p), Some(c)) = (parent_idx, child_idx) {
            out.push_str(&format!(
                "  N{} -->|{:+.2}| N{}\n",
                p, edge.score_improvement, c
            ));
        }
    }
    out.push_str("```\n");
    out
}

fn deterministic_drill_decision(
    parent_step: &InvestigationStep,
    mover: &InvestigationMover,
    mode: InvestigationMode,
    next_dimension: &str,
) -> String {
    let detail = match mode {
        InvestigationMode::ConcentrationDrivers => format!(
            "delta share {} pp, score {}",
            signed_fmt_num(mover.delta_share_pp, 2),
            fmt_num(mover_score(mover, mode), 2)
        ),
        InvestigationMode::ChangeDrivers
        | InvestigationMode::CompareSnapshots
        | InvestigationMode::RecommendNext => format!(
            "delta {} {}, score {}",
            parent_step.primary_metric,
            signed_fmt_num(mover.delta_primary_metric_value, 2),
            fmt_num(mover_score(mover, mode), 2)
        ),
    };
    format!(
        "selected {}='{}' ({}) and drilled by {}",
        parent_step.dimension, mover.segment, detail, next_dimension
    )
}

fn choose_next_dimension(
    drill_dimensions: &[String],
    used_dimensions: &[String],
) -> Option<String> {
    drill_dimensions
        .iter()
        .find(|d| !used_dimensions.iter().any(|u| u == *d))
        .cloned()
}

fn remaining_drill_dimensions(
    drill_dimensions: &[String],
    used_dimensions: &[String],
) -> Vec<String> {
    drill_dimensions
        .iter()
        .filter(|d| !used_dimensions.iter().any(|u| u == *d))
        .cloned()
        .collect::<Vec<_>>()
}

fn investigation_mode_label(mode: InvestigationMode) -> &'static str {
    match mode {
        InvestigationMode::ChangeDrivers => "change_drivers",
        InvestigationMode::ConcentrationDrivers => "concentration_drivers",
        InvestigationMode::CompareSnapshots => "compare_snapshots",
        InvestigationMode::RecommendNext => "recommend_next",
    }
}

fn mode_from_arg(mode: InvestigationModeArg) -> InvestigationMode {
    match mode {
        InvestigationModeArg::ChangeDrivers => InvestigationMode::ChangeDrivers,
        InvestigationModeArg::ConcentrationDrivers => InvestigationMode::ConcentrationDrivers,
        InvestigationModeArg::CompareSnapshots => InvestigationMode::CompareSnapshots,
        InvestigationModeArg::RecommendNext => InvestigationMode::RecommendNext,
    }
}

fn resolve_investigation_mode(args: &InvestigateWorkflowArgs) -> InvestigationMode {
    if let Some(mode) = args.mode {
        return mode_from_arg(mode);
    }
    route_investigation_mode(&args.question)
}

fn route_investigation_mode(question: &str) -> InvestigationMode {
    let q = question.to_lowercase();
    if q.contains("concentration") || q.contains("concentrated") {
        return InvestigationMode::ConcentrationDrivers;
    }
    if q.contains("what should i inspect")
        || q.contains("what should we inspect")
        || q.contains("what next")
        || q.contains("inspect next")
    {
        return InvestigationMode::RecommendNext;
    }
    if q.contains("compare")
        || q.contains("biggest shift")
        || q.contains("biggest change")
        || q.contains("largest change")
    {
        return InvestigationMode::CompareSnapshots;
    }
    InvestigationMode::ChangeDrivers
}

fn investigate_input_kind_label(kind: InvestigateInputKind) -> &'static str {
    match kind {
        InvestigateInputKind::JsonArtifacts => "artifacts_json",
        InvestigateInputKind::CsvDatasets => "datasets_csv",
    }
}

fn default_investigate_workflow_out(args: &InvestigateWorkflowArgs) -> PathBuf {
    let base = PathBuf::from("artifacts/investigate_workflow");
    match args.output_format {
        InvestigateOutputFormat::Md | InvestigateOutputFormat::Both => base.with_extension("md"),
        InvestigateOutputFormat::Json => base.with_extension("json"),
    }
}

fn recommended_next_question(
    mode: InvestigationMode,
    last_step: Option<&InvestigationStep>,
) -> String {
    match (mode, last_step, last_step.and_then(|s| s.movers.first())) {
        (_, Some(step), Some(mover))
            if mover.segment != "(blank)" && mover_score(mover, mode).abs() > 0.005 =>
        {
            let scope_prefix = if step.scope.is_empty() {
                String::new()
            } else {
                format!("within scope [{}], ", format_scope(&step.scope))
            };
            let delta_detail = match mode {
                InvestigationMode::ConcentrationDrivers => {
                    format!("delta share {} pp", signed_fmt_num(mover.delta_share_pp, 2))
                }
                InvestigationMode::ChangeDrivers
                | InvestigationMode::CompareSnapshots
                | InvestigationMode::RecommendNext => format!(
                    "delta {} {}",
                    step.primary_metric,
                    signed_fmt_num(mover.delta_primary_metric_value, 2)
                ),
            };
            format!(
                "Drill deeper {}on {}='{}' to explain {} and whether it is persistent.",
                scope_prefix, step.dimension, mover.segment, delta_detail
            )
        }
        (InvestigationMode::ConcentrationDrivers, _, _) => {
            "Which segment-level policy or pricing action can reduce top-5 concentration next period?"
                .to_string()
        }
        (InvestigationMode::CompareSnapshots, _, _) => {
            "Which of these movers is expected to remain material in the next snapshot?".to_string()
        }
        (InvestigationMode::RecommendNext, _, _) | (InvestigationMode::ChangeDrivers, _, _) => {
            "Which segment should we test first to reverse the observed change?".to_string()
        }
    }
}

struct InvestigationInputLabels<'a> {
    base: &'a str,
    new: &'a str,
}

#[derive(Debug, Clone)]
struct InvestigationBranchSummaryRow {
    root_dimension: String,
    root_segment: String,
    root_score: f64,
    deepest_depth: usize,
    deepest_dimension: String,
    deepest_segment: String,
    deepest_delta_primary_metric_value: f64,
    deepest_score: f64,
}

fn build_branch_summary_rows(
    steps: &[InvestigationStep],
    mode: InvestigationMode,
    limit: usize,
) -> Vec<InvestigationBranchSummaryRow> {
    if steps.len() <= 1 || limit == 0 {
        return Vec::new();
    }

    let Some(_root_step) = steps.first() else {
        return Vec::new();
    };
    let mut root_score_by_dim_segment = HashMap::<(String, String), f64>::new();
    for step in steps.iter().filter(|s| s.scope.is_empty()) {
        for mover in &step.movers {
            let key = (step.dimension.clone(), mover.segment.clone());
            let score = mover_score(mover, mode);
            let entry = root_score_by_dim_segment.entry(key).or_insert(score);
            if score > *entry {
                *entry = score;
            }
        }
    }

    let mut rows = HashMap::<String, InvestigationBranchSummaryRow>::new();
    for step in steps.iter().skip(1) {
        let Some((first_dim, first_segment)) = step.scope.first() else {
            continue;
        };
        let top = step.movers.first();
        let step_score = top.map(|m| mover_score(m, mode)).unwrap_or(0.0);
        let step_segment = top
            .map(|m| m.segment.clone())
            .unwrap_or_else(|| "(none)".to_string());
        let step_delta = top.map(|m| m.delta_primary_metric_value).unwrap_or(0.0);
        let key = format!("{}={}", first_dim, first_segment);
        let root_score = root_score_by_dim_segment
            .get(&(first_dim.clone(), first_segment.clone()))
            .copied()
            .unwrap_or(step_score);

        let candidate = InvestigationBranchSummaryRow {
            root_dimension: first_dim.clone(),
            root_segment: first_segment.clone(),
            root_score,
            deepest_depth: step.depth,
            deepest_dimension: step.dimension.clone(),
            deepest_segment: step_segment,
            deepest_delta_primary_metric_value: step_delta,
            deepest_score: step_score,
        };

        match rows.get(&key) {
            None => {
                rows.insert(key, candidate);
            }
            Some(existing) => {
                let should_replace = candidate.deepest_depth > existing.deepest_depth
                    || (candidate.deepest_depth == existing.deepest_depth
                        && candidate.deepest_score > existing.deepest_score + f64::EPSILON);
                if should_replace {
                    rows.insert(key, candidate);
                }
            }
        }
    }

    let mut out = rows.into_values().collect::<Vec<_>>();
    out.sort_by(|a, b| {
        b.root_score
            .partial_cmp(&a.root_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.deepest_score
                    .partial_cmp(&a.deepest_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.root_segment.cmp(&b.root_segment))
    });
    out.truncate(limit);
    out
}

struct InvestigationRenderData<'a> {
    major_global_changes: &'a [InvestigationMajorChange],
    coverage: &'a InvestigationCoverage,
    branch_graph: &'a InvestigationBranchGraph,
    steps: &'a [InvestigationStep],
    trace: &'a [InvestigationTraceStep],
    stop_reason: &'a str,
    recommended_next_question: &'a str,
}

fn render_investigation_workflow_markdown(
    args: &InvestigateWorkflowArgs,
    input_labels: &InvestigationInputLabels<'_>,
    mode: InvestigationMode,
    data: &InvestigationRenderData<'_>,
) -> String {
    let md_cell = |v: &str| v.replace('|', "\\|").replace('\n', " ");
    let mut md = String::new();
    md.push_str("# Investigation Report\n\n");
    md.push_str(&format!("- Question: {}\n", args.question));
    md.push_str(&format!("- Mode: {}\n", investigation_mode_label(mode)));
    md.push_str(&format!("- Base input: {}\n", input_labels.base));
    md.push_str(&format!("- New input: {}\n", input_labels.new));
    md.push('\n');

    if let Some(step0) = data.steps.first() {
        md.push_str("## Top-level finding\n\n");
        let base_total = step0
            .movers
            .iter()
            .map(|m| m.base_primary_metric_value)
            .sum::<f64>();
        let new_total = step0
            .movers
            .iter()
            .map(|m| m.new_primary_metric_value)
            .sum::<f64>();
        let delta_total = new_total - base_total;
        if step0.movers.len() == step0.segment_count {
            md.push_str(&format!(
                "- Between compared periods, total `{}` changed from {} to {} (delta = {}).\n",
                step0.primary_metric,
                fmt_num(base_total, 2),
                fmt_num(new_total, 2),
                signed_fmt_num(delta_total, 2)
            ));
        } else {
            md.push_str(&format!(
                "- Across reported movers only ({} of {} segments), subtotal `{}` changed from {} to {} (delta = {}); increase `--top-movers` for full-period total.\n",
                step0.movers.len(),
                step0.segment_count,
                step0.primary_metric,
                fmt_num(base_total, 2),
                fmt_num(new_total, 2),
                signed_fmt_num(delta_total, 2)
            ));
        }
        if let Some(top) = step0.movers.first() {
            md.push_str(&format!(
                "- On `{}` the strongest mover is `{}` (delta {} = {}, delta share = {:+.2} pp).\n",
                step0.dimension,
                top.segment,
                step0.primary_metric,
                fmt_num(top.delta_primary_metric_value, 2),
                top.delta_share_pp
            ));
        } else {
            md.push_str("- No movers were available at top level.\n");
        }
        let base_metric_total = step0
            .movers
            .iter()
            .map(|m| m.base_primary_metric_value.max(0.0))
            .sum::<f64>();
        let new_metric_total = step0
            .movers
            .iter()
            .map(|m| m.new_primary_metric_value.max(0.0))
            .sum::<f64>();
        let metric_share = |value: f64, total: f64| -> f64 {
            if total <= 0.0 {
                0.0
            } else {
                (value.max(0.0) / total) * 100.0
            }
        };
        let mut metric_sorted = step0.movers.iter().collect::<Vec<_>>();
        metric_sorted.sort_by(|a, b| {
            b.new_primary_metric_value
                .partial_cmp(&a.new_primary_metric_value)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.segment.cmp(&b.segment))
        });
        let base_top1_metric_pct = step0
            .movers
            .iter()
            .map(|m| metric_share(m.base_primary_metric_value, base_metric_total))
            .fold(0.0, f64::max);
        let new_top1_metric_pct = step0
            .movers
            .iter()
            .map(|m| metric_share(m.new_primary_metric_value, new_metric_total))
            .fold(0.0, f64::max);
        let base_top5_metric_pct = metric_sorted
            .iter()
            .take(5)
            .map(|m| metric_share(m.base_primary_metric_value, base_metric_total))
            .sum::<f64>();
        let new_top5_metric_pct = metric_sorted
            .iter()
            .take(5)
            .map(|m| metric_share(m.new_primary_metric_value, new_metric_total))
            .sum::<f64>();

        if step0.segment_count <= 5 {
            md.push_str(&format!(
                "- Top-5 record-share concentration is saturated at 100.00% because only {} segments exist; top-1 record-share concentration moved from {:.2}% to {:.2}% ({:+.2} pp).\n",
                step0.segment_count,
                step0.top1_concentration_base_pct,
                step0.top1_concentration_new_pct,
                step0.top1_concentration_delta_pp
            ));
        } else {
            md.push_str(&format!(
                "- Top-5 metric-share concentration moved from {:.2}% to {:.2}% ({:+.2} pp).\n",
                base_top5_metric_pct,
                new_top5_metric_pct,
                new_top5_metric_pct - base_top5_metric_pct
            ));
        }
        md.push_str(
            "- Concentration tables include both metric-share (`primary metric`) and record-share (`rows`) for this grouping dimension.\n",
        );
        md.push('\n');
        md.push_str("### Concentration snapshot\n\n");
        md.push_str("| Measure | Base | New | Delta |\n");
        md.push_str("|---|---:|---:|---:|\n");
        md.push_str(&format!(
            "| Top-1 concentration (metric-share) | {:.2}% | {:.2}% | {:+.2} pp |\n",
            base_top1_metric_pct,
            new_top1_metric_pct,
            new_top1_metric_pct - base_top1_metric_pct
        ));
        md.push_str(&format!(
            "| Top-5 concentration (metric-share) | {:.2}% | {:.2}% | {:+.2} pp |\n",
            base_top5_metric_pct,
            new_top5_metric_pct,
            new_top5_metric_pct - base_top5_metric_pct
        ));
        md.push_str(&format!(
            "| Top-1 concentration (record-share) | {:.2}% | {:.2}% | {:+.2} pp |\n",
            step0.top1_concentration_base_pct,
            step0.top1_concentration_new_pct,
            step0.top1_concentration_delta_pp
        ));
        md.push_str(&format!(
            "| Top-5 concentration (record-share) | {:.2}% | {:.2}% | {:+.2} pp |\n",
            step0.top5_concentration_base_pct,
            step0.top5_concentration_new_pct,
            step0.top5_concentration_delta_pp
        ));
        md.push('\n');

        if !step0.movers.is_empty() {
            let mut largest_segments = step0.movers.iter().collect::<Vec<_>>();
            largest_segments.sort_by(|a, b| {
                b.new_primary_metric_value
                    .partial_cmp(&a.new_primary_metric_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.segment.cmp(&b.segment))
            });
            let top5_names = largest_segments
                .iter()
                .take(5)
                .map(|m| {
                    format!(
                        "{} ({:.2}% metric share)",
                        m.segment,
                        metric_share(m.new_primary_metric_value, new_metric_total)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            md.push_str(&format!(
                "- Largest new-period segments by metric share: {}.\n",
                if top5_names.is_empty() {
                    "(none)".to_string()
                } else {
                    top5_names
                }
            ));
            md.push('\n');

            md.push_str("### Segments behind concentration (largest by metric share)\n\n");
            md.push_str("| # | Segment | Base metric share | New metric share | Delta metric share | New record share |\n");
            md.push_str("|---:|---|---:|---:|---:|---:|\n");
            for (idx, mover) in largest_segments.iter().take(5).enumerate() {
                md.push_str(&format!(
                    "| {} | {} | {:.2}% | {:.2}% | {:+.2} pp | {:.2}% |\n",
                    idx + 1,
                    md_cell(&mover.segment),
                    metric_share(mover.base_primary_metric_value, base_metric_total),
                    metric_share(mover.new_primary_metric_value, new_metric_total),
                    metric_share(mover.new_primary_metric_value, new_metric_total)
                        - metric_share(mover.base_primary_metric_value, base_metric_total),
                    mover.new_share_pct
                ));
            }
            if step0.segment_count > step0.movers.len() {
                md.push_str(&format!(
                    "\n_Note: concentration table is derived from reported movers only ({} of {} segments in this scope; tune `--top-movers` to expand)._\
\n",
                    step0.movers.len(),
                    step0.segment_count
                ));
            }
            md.push('\n');

            md.push_str("### Top-level movers (top 5)\n\n");
            md.push_str("| # | Segment | Base share | New share | Delta share | Delta metric |\n");
            md.push_str("|---:|---|---:|---:|---:|---:|\n");
            for (idx, mover) in step0.movers.iter().take(5).enumerate() {
                md.push_str(&format!(
                    "| {} | {} | {:.2}% | {:.2}% | {:+.2} pp | {} |\n",
                    idx + 1,
                    md_cell(&mover.segment),
                    mover.base_share_pct,
                    mover.new_share_pct,
                    mover.delta_share_pp,
                    signed_fmt_num(mover.delta_primary_metric_value, 2)
                ));
            }
            if step0.segment_count > step0.movers.len() {
                md.push_str(&format!(
                    "\n_Note: showing {} movers ({} total segments in this scope; tune `--top-movers` to expand)._\
\n",
                    step0.movers.len(),
                    step0.segment_count
                ));
            }
            md.push('\n');
        }

        if !data.major_global_changes.is_empty() {
            md.push_str("### Strongest mover by configured dimension\n\n");
            md.push_str("| # | Dimension | Segment | Delta metric | Delta share |\n");
            md.push_str("|---:|---|---|---:|---:|\n");
            for (idx, change) in data.major_global_changes.iter().enumerate() {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {:+.2} pp |\n",
                    idx + 1,
                    md_cell(&change.dimension),
                    md_cell(&change.segment),
                    signed_fmt_num(change.delta_primary_metric_value, 2),
                    change.delta_share_pp
                ));
            }
            md.push('\n');
        }
        md.push('\n');
    }

    md.push_str("## Follow-up findings\n\n");
    if data.steps.len() <= 1 {
        md.push_str("- No follow-up step executed.\n");
    } else {
        for step in data.steps.iter().skip(1) {
            if let Some(top) = step.movers.first() {
                let scope_txt = if step.scope.is_empty() {
                    "global".to_string()
                } else {
                    step.scope
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                if step.scope.is_empty() {
                    md.push_str(&format!(
                        "- Global follow-up grouped by `{}`: strongest mover `{}` (delta {} = {}, delta share = {:+.2} pp).\n",
                        step.dimension,
                        top.segment,
                        step.primary_metric,
                        fmt_num(top.delta_primary_metric_value, 2),
                        top.delta_share_pp
                    ));
                } else {
                    md.push_str(&format!(
                        "- Depth {} scope [{}] grouped by `{}`: strongest mover `{}` (delta {} = {}, delta share = {:+.2} pp).\n",
                        step.depth,
                        scope_txt,
                        step.dimension,
                        top.segment,
                        step.primary_metric,
                        fmt_num(top.delta_primary_metric_value, 2),
                        top.delta_share_pp
                    ));
                }
            }
        }
        md.push('\n');
        md.push_str("### Follow-up strongest movers table\n\n");
        md.push_str(
            "| Depth | Scope | Grouped by | Strongest mover | Delta metric | Delta share |\n",
        );
        md.push_str("|---:|---|---|---|---:|---:|\n");
        for step in data.steps.iter().skip(1) {
            if let Some(top) = step.movers.first() {
                let scope_txt = if step.scope.is_empty() {
                    "global".to_string()
                } else {
                    step.scope
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {:+.2} pp |\n",
                    step.depth,
                    md_cell(&scope_txt),
                    md_cell(&step.dimension),
                    md_cell(&top.segment),
                    signed_fmt_num(top.delta_primary_metric_value, 2),
                    top.delta_share_pp
                ));
            }
        }
    }
    md.push('\n');

    let branch_summary = build_branch_summary_rows(data.steps, mode, 3);
    if !branch_summary.is_empty() {
        md.push_str("### Top branches (side-by-side)\n\n");
        md.push_str("| Branch | Root scope | Root score | Deepest step | Deepest strongest mover | Deepest delta metric |\n");
        md.push_str("|---:|---|---:|---|---|---:|\n");
        for (idx, row) in branch_summary.iter().enumerate() {
            md.push_str(&format!(
                "| {} | {}={} | {} | depth {} / {} | {} | {} |\n",
                idx + 1,
                md_cell(&row.root_dimension),
                md_cell(&row.root_segment),
                fmt_num(row.root_score, 2),
                row.deepest_depth,
                md_cell(&row.deepest_dimension),
                md_cell(&row.deepest_segment),
                signed_fmt_num(row.deepest_delta_primary_metric_value, 2)
            ));
        }
        md.push('\n');
    }

    md.push_str("## Coverage\n\n");
    md.push_str(&format!(
        "- Total delta (`{}`) = {} (abs {}).\n",
        data.steps
            .first()
            .map(|s| s.primary_metric.as_str())
            .unwrap_or("primary_metric"),
        signed_fmt_num(data.coverage.top_level_total_delta, 2),
        fmt_num(data.coverage.total_delta_abs, 2)
    ));
    if let Some(segment) = &data.coverage.top_level_strongest_segment {
        md.push_str(&format!(
            "- Top-level strongest segment `{}` explains {:.2}% of total delta.\n",
            segment, data.coverage.top_level_strongest_explained_pct
        ));
    }
    if !data.coverage.step_coverage.is_empty() {
        md.push_str("\n### Step coverage table\n\n");
        md.push_str("| Step | Depth | Dimension | Strongest segment | Delta metric | Explained % of total | Residual abs after step |\n");
        md.push_str("|---:|---:|---|---|---:|---:|---:|\n");
        for c in &data.coverage.step_coverage {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {:.2}% | {} |\n",
                c.step_index,
                c.depth,
                md_cell(&c.dimension),
                md_cell(&c.strongest_segment),
                signed_fmt_num(c.strongest_delta_primary_metric_value, 2),
                c.strongest_explained_pct_of_total_delta,
                fmt_num(c.residual_delta_abs_after_step, 2)
            ));
        }
        md.push('\n');
    }

    if !data.branch_graph.nodes.is_empty() {
        md.push_str("## Drill-down graph\n\n");
        md.push_str(&render_branch_graph_mermaid(data.branch_graph));
        md.push('\n');
    }

    md.push_str("## Why it stopped\n\n");
    md.push_str(&format!("- {}\n\n", data.stop_reason));

    md.push_str("## Recommended next question\n\n");
    md.push_str(&format!("- {}\n\n", data.recommended_next_question));

    md.push_str("## Decision trace\n\n");
    for t in data.trace {
        let scope_txt = if t.scope.is_empty() {
            "global".to_string()
        } else {
            t.scope
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(", ")
        };
        md.push_str(&format!(
            "- depth={} action={} scope=[{}] decision={}\n",
            t.depth, t.action, scope_txt, t.decision
        ));
    }
    md
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
        let share = g
            .get("count_share_pct")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        let primary = value_by_key_case_insensitive(&g, primary_metric)
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        out.insert(name, (count, share, primary));
    }
    out
}

fn value_by_key_case_insensitive<'a>(
    v: &'a serde_json::Value,
    key: &str,
) -> Option<&'a serde_json::Value> {
    if let Some(exact) = v.get(key) {
        return Some(exact);
    }
    let obj = v.as_object()?;
    obj.iter().find_map(|(k, val)| {
        if k.eq_ignore_ascii_case(key) {
            Some(val)
        } else {
            None
        }
    })
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
            let cert = cert
                .map_err(|e| anyhow!("failed to parse PEM cert in '{}': {}", path.display(), e))?;
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

fn decode_investigate_config_entry(
    path: &Path,
    section: toml::Value,
    context: &str,
) -> Result<InvestigateConfigEntry> {
    if !section.is_table() {
        return Err(anyhow!(
            "investigate config '{}' {} must be a TOML table",
            path.display(),
            context
        ));
    }
    section.try_into().map_err(|e| {
        anyhow!(
            "failed to decode investigate config '{}' {}: {}",
            path.display(),
            context,
            e
        )
    })
}

fn load_investigate_config(path: &Path) -> Result<InvestigateConfigEntry> {
    let text = fs::read_to_string(path).map_err(|e| {
        anyhow!(
            "failed to read investigate config '{}': {}",
            path.display(),
            e
        )
    })?;
    let raw: toml::Value = toml::from_str(&text).map_err(|e| {
        anyhow!(
            "failed to parse investigate config '{}': {}",
            path.display(),
            e
        )
    })?;
    if let Some(section) = raw.get("investigate").cloned() {
        return decode_investigate_config_entry(path, section, "from [investigate]");
    }
    if let Some(profiles) = raw.get("profiles").and_then(|v| v.as_table()) {
        if profiles.is_empty() {
            return Err(anyhow!(
                "investigate config '{}' has empty [profiles] table",
                path.display()
            ));
        }
        if profiles.len() > 1 {
            return Err(anyhow!(
                "investigate config '{}' has multiple [profiles.*] entries; choose one with --profile <name> --profile-config <path.toml>",
                path.display()
            ));
        }
        let (name, section) = profiles.iter().next().expect("checked non-empty");
        return decode_investigate_config_entry(
            path,
            section.clone(),
            &format!("from [profiles.{}]", name),
        );
    }
    decode_investigate_config_entry(path, raw, "from root table")
}

fn load_investigate_profile(path: &Path, profile_raw: &str) -> Result<InvestigateConfigEntry> {
    let text = fs::read_to_string(path).map_err(|e| {
        anyhow!(
            "failed to read investigate profile config '{}': {}",
            path.display(),
            e
        )
    })?;
    let raw: toml::Value = toml::from_str(&text).map_err(|e| {
        anyhow!(
            "failed to parse investigate profile config '{}': {}",
            path.display(),
            e
        )
    })?;
    let profiles = raw
        .get("profiles")
        .and_then(|v| v.as_table())
        .ok_or_else(|| {
            anyhow!(
                "investigate profile config '{}' must contain [profiles.<name>] entries",
                path.display()
            )
        })?;
    let profile = profile_raw.trim();
    let (matched_name, section) = profiles
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(profile))
        .ok_or_else(|| {
            anyhow!(
                "investigate profile '{}' not found in {}",
                profile_raw,
                path.display()
            )
        })?;
    decode_investigate_config_entry(
        path,
        section.clone(),
        &format!("for profile '{}'", matched_name),
    )
}

fn parse_mode_arg_from_str(raw: &str) -> Result<InvestigationModeArg> {
    match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "change_drivers" => Ok(InvestigationModeArg::ChangeDrivers),
        "concentration_drivers" => Ok(InvestigationModeArg::ConcentrationDrivers),
        "compare_snapshots" => Ok(InvestigationModeArg::CompareSnapshots),
        "recommend_next" => Ok(InvestigationModeArg::RecommendNext),
        _ => Err(anyhow!(
            "invalid investigate mode '{}'; expected one of: change_drivers, concentration_drivers, compare_snapshots, recommend_next",
            raw
        )),
    }
}

fn parse_planner_from_str(raw: &str) -> Result<InvestigationPlanner> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "deterministic" => Ok(InvestigationPlanner::Deterministic),
        "llm" => Ok(InvestigationPlanner::Llm),
        _ => Err(anyhow!(
            "invalid planner '{}'; expected deterministic or llm",
            raw
        )),
    }
}

fn parse_backend_arg_from_str(raw: &str) -> Result<BackendArg> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "local" => Ok(BackendArg::Local),
        "bedrock" => Ok(BackendArg::Bedrock),
        _ => Err(anyhow!(
            "invalid planner backend '{}'; expected local or bedrock",
            raw
        )),
    }
}

fn parse_investigate_output_format_from_str(raw: &str) -> Result<InvestigateOutputFormat> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "md" => Ok(InvestigateOutputFormat::Md),
        "json" => Ok(InvestigateOutputFormat::Json),
        "both" => Ok(InvestigateOutputFormat::Both),
        _ => Err(anyhow!(
            "invalid investigate output format '{}'; expected md, json, or both",
            raw
        )),
    }
}

fn parse_postgres_ssl_mode_from_str(raw: &str) -> Result<PostgresSslMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "disable" => Ok(PostgresSslMode::Disable),
        "prefer" => Ok(PostgresSslMode::Prefer),
        "require" => Ok(PostgresSslMode::Require),
        _ => Err(anyhow!(
            "invalid postgres ssl mode '{}'; expected disable, prefer, or require",
            raw
        )),
    }
}

fn parse_time_grain_from_str(raw: &str) -> Result<TimeGrain> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "day" => Ok(TimeGrain::Day),
        "week" => Ok(TimeGrain::Week),
        "month" => Ok(TimeGrain::Month),
        "year" => Ok(TimeGrain::Year),
        _ => Err(anyhow!(
            "invalid time grain '{}'; expected day, week, month, or year",
            raw
        )),
    }
}

fn parse_period_preset_from_str(raw: &str) -> Result<PeriodPreset> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "current" => Ok(PeriodPreset::Current),
        "previous" => Ok(PeriodPreset::Previous),
        "last" => Ok(PeriodPreset::Last),
        _ => Err(anyhow!(
            "invalid period '{}'; expected current, previous, or last",
            raw
        )),
    }
}

fn apply_investigate_config(mut args: InvestigateWorkflowArgs) -> Result<InvestigateWorkflowArgs> {
    if args.config.is_some() && args.profile.is_some() {
        return Err(anyhow!(
            "use either --config <path.toml> or --profile <name> [--profile-config <path.toml>], not both"
        ));
    }
    if args.config.is_some() && args.profile_config.is_some() {
        return Err(anyhow!(
            "use either --config <path.toml> or --profile <name> [--profile-config <path.toml>], not both"
        ));
    }
    if args.profile_config.is_some() && args.profile.is_none() {
        return Err(anyhow!("--profile-config requires --profile <name>"));
    }

    let cfg_opt = if let Some(cfg_path) = args.config.clone() {
        Some(load_investigate_config(&cfg_path)?)
    } else if let Some(profile_raw) = args.profile.clone() {
        if let Some(cfg_path) = args.profile_config.clone() {
            Some(load_investigate_profile(&cfg_path, &profile_raw)?)
        } else {
            let profile_trimmed = profile_raw.trim();
            let looks_like_path = profile_trimmed.contains(std::path::MAIN_SEPARATOR)
                || profile_trimmed.contains('/')
                || profile_trimmed.contains('\\')
                || profile_trimmed.ends_with(".toml");
            if looks_like_path {
                Some(load_investigate_config(Path::new(profile_trimmed))?)
            } else {
                return Err(anyhow!(
                    "investigate --profile '{}' expects --profile-config <path.toml>; or use --config <path.toml> for single investigate configs",
                    profile_raw
                ));
            }
        }
    } else {
        None
    };

    if let Some(cfg) = cfg_opt {
        if args.question.trim().is_empty() {
            if let Some(v) = cfg.question {
                args.question = v;
            }
        }
        if args.mode.is_none() {
            if let Some(v) = cfg.mode.as_deref() {
                args.mode = Some(parse_mode_arg_from_str(v)?);
            }
        }
        if args.base.is_none() {
            if let Some(v) = cfg.base {
                args.base = Some(PathBuf::from(v));
            }
        }
        if args.new.is_none() {
            if let Some(v) = cfg.new {
                args.new = Some(PathBuf::from(v));
            }
        }
        if args.postgres_url.is_none() {
            args.postgres_url = cfg.postgres_url;
        }
        if args.postgres_ssl_mode == PostgresSslMode::Prefer {
            if let Some(v) = cfg.postgres_ssl_mode.as_deref() {
                args.postgres_ssl_mode = parse_postgres_ssl_mode_from_str(v)?;
            }
        }
        if args.postgres_ca_file.is_none() {
            if let Some(v) = cfg.postgres_ca_file {
                args.postgres_ca_file = Some(PathBuf::from(v));
            }
        }
        if args.query.is_none() {
            args.query = cfg.query;
        }
        if args.query_file.is_none() {
            if let Some(v) = cfg.query_file {
                args.query_file = Some(PathBuf::from(v));
            }
        }
        if args.metric.is_none() {
            args.metric = cfg.metric;
        }
        if args.date_column.is_none() {
            args.date_column = cfg.date_column;
        }
        if args.time_grain.is_none() {
            if let Some(v) = cfg.time_grain.as_deref() {
                args.time_grain = Some(parse_time_grain_from_str(v)?);
            }
        }
        if args.period.is_none() {
            if let Some(v) = cfg.period.as_deref() {
                args.period = Some(parse_period_preset_from_str(v)?);
            }
        }
        if args.anchor_date.is_none() {
            args.anchor_date = cfg.anchor_date;
        }
        if args.current_start.is_none() {
            args.current_start = cfg.current_start;
        }
        if args.current_end.is_none() {
            args.current_end = cfg.current_end;
        }
        if args.previous_start.is_none() {
            args.previous_start = cfg.previous_start;
        }
        if args.previous_end.is_none() {
            args.previous_end = cfg.previous_end;
        }
        if args.dimensions.is_empty() {
            if let Some(v) = cfg.dimensions {
                args.dimensions = v;
            }
        }
        if args.drill_fields.is_empty() {
            if let Some(v) = cfg.drill_fields {
                args.drill_fields = v;
            }
        }
        if args.max_depth == 2 {
            if let Some(v) = cfg.max_depth {
                args.max_depth = v;
            }
        }
        if args.max_branches == 3 {
            if let Some(v) = cfg.max_branches {
                args.max_branches = v;
            }
        }
        if (args.min_contribution - 5.0).abs() <= f64::EPSILON {
            if let Some(v) = cfg.min_contribution {
                args.min_contribution = v;
            }
        }
        if args.min_delta_abs.abs() <= f64::EPSILON {
            if let Some(v) = cfg.min_delta_abs {
                args.min_delta_abs = v;
            }
        }
        if args.min_score_improvement.abs() <= f64::EPSILON {
            if let Some(v) = cfg.min_score_improvement {
                args.min_score_improvement = v;
            }
        }
        if args.min_slice_rows == 5 {
            if let Some(v) = cfg.min_slice_rows {
                args.min_slice_rows = v;
            }
        }
        if args.top_movers == 12 {
            if let Some(v) = cfg.top_movers {
                args.top_movers = v;
            }
        }
        if args.planner == InvestigationPlanner::Deterministic {
            if let Some(v) = cfg.planner.as_deref() {
                args.planner = parse_planner_from_str(v)?;
            }
        }
        if args.planner_backend == BackendArg::Local {
            if let Some(v) = cfg.planner_backend.as_deref() {
                args.planner_backend = parse_backend_arg_from_str(v)?;
            }
        }
        if args.planner_model.is_none() {
            args.planner_model = cfg.planner_model;
        }
        if !args.verbose {
            if let Some(v) = cfg.verbose {
                args.verbose = v;
            }
        }
        if !args.trace {
            if let Some(v) = cfg.trace {
                args.trace = v;
            }
        }
        if args.output_format == InvestigateOutputFormat::Both {
            if let Some(v) = cfg.output_format.as_deref() {
                args.output_format = parse_investigate_output_format_from_str(v)?;
            }
        }
        if args.out.is_none() {
            if let Some(v) = cfg.out {
                args.out = Some(PathBuf::from(v));
            }
        }
    }

    args.dimensions = dedup_csv_fields(&args.dimensions);
    args.drill_fields = dedup_csv_fields(&args.drill_fields);

    Ok(args)
}

fn dedup_csv_fields(values: &[String]) -> Vec<String> {
    let mut out = Vec::<String>::new();
    let mut seen = HashSet::<String>::new();
    for raw in values {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let key = trimmed.to_ascii_lowercase();
        if seen.insert(key) {
            out.push(trimmed.to_string());
        }
    }
    out
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
    artifacts: &Path,
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
    artifacts: &Path,
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

    type WorstAttribution = (String, f64, Vec<(String, f64)>);
    let mut worst: Option<WorstAttribution> = None;
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

fn row_get(row: &StringRecord, idx: usize) -> Result<&str> {
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
            .filter(|(t, _)| !low_list.contains(&t.as_str()))
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

#[derive(Debug, Deserialize, Default)]
struct InvestigateConfigEntry {
    question: Option<String>,
    mode: Option<String>,
    base: Option<String>,
    new: Option<String>,
    postgres_url: Option<String>,
    postgres_ssl_mode: Option<String>,
    postgres_ca_file: Option<String>,
    query: Option<String>,
    query_file: Option<String>,
    metric: Option<String>,
    date_column: Option<String>,
    time_grain: Option<String>,
    period: Option<String>,
    anchor_date: Option<String>,
    current_start: Option<String>,
    current_end: Option<String>,
    previous_start: Option<String>,
    previous_end: Option<String>,
    dimensions: Option<Vec<String>>,
    drill_fields: Option<Vec<String>>,
    max_depth: Option<usize>,
    max_branches: Option<usize>,
    min_contribution: Option<f64>,
    min_delta_abs: Option<f64>,
    min_score_improvement: Option<f64>,
    min_slice_rows: Option<u64>,
    top_movers: Option<usize>,
    planner: Option<String>,
    planner_backend: Option<String>,
    planner_model: Option<String>,
    verbose: Option<bool>,
    trace: Option<bool>,
    output_format: Option<String>,
    out: Option<String>,
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

#[allow(clippy::too_many_arguments)]
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
        let gk = compose_group_key(&rec, &group_idxs, &resolved_groups, normalize_text_groups);
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
        if count_only {
            "Count-only (no numeric metrics)"
        } else {
            agg.label()
        }
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
        movers.sort_by(|a, b| a.5.partial_cmp(&b.5).unwrap_or(std::cmp::Ordering::Equal));
        let mut concentration_movers = period_by_group
            .iter()
            .map(|(g, (cc, cv, pc, pv))| {
                let base_share = pct(*pc, period_totals.2);
                let new_share = pct(*cc, period_totals.0);
                let d_share = new_share - base_share;
                (
                    g.clone(),
                    *pc,
                    *cc,
                    base_share,
                    new_share,
                    d_share,
                    *pv,
                    *cv,
                )
            })
            .collect::<Vec<_>>();
        concentration_movers.sort_by(|a, b| {
            b.5.abs()
                .partial_cmp(&a.5.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let prev_seg_count = period_by_group
            .values()
            .filter(|(_, _, pc, _)| *pc > 0)
            .count();
        let curr_seg_count = period_by_group
            .values()
            .filter(|(cc, _, _, _)| *cc > 0)
            .count();
        let prev_top5_count = concentration_movers
            .iter()
            .take(5)
            .map(|x| x.1)
            .sum::<u64>();
        let curr_top5_count = concentration_movers
            .iter()
            .take(5)
            .map(|x| x.2)
            .sum::<u64>();
        let prev_top5_pct = pct(prev_top5_count, period_totals.2);
        let curr_top5_pct = pct(curr_top5_count, period_totals.0);

        md.push_str("## Period Comparison\n\n");
        md.push_str(&format!("- Date column: `{}`\n", cfg.date_column));
        if let (Some(g), Some(p)) = (cfg.time_grain, cfg.period) {
            md.push_str(&format!("- Window mode: {:?} / {:?}\n", g, p));
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
        for (i, (g, pc, cc, _bs, _ns, d_share, _pv, _cv)) in
            concentration_movers.iter().take(5).enumerate()
        {
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

        by_count_share.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
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
            by_per_record
                .sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap_or(std::cmp::Ordering::Equal));
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
    md.push('|');
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
    md.push('|');
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
            if (2..=60).contains(&card)
                && fill_ratio > 0.2
                && !looks_like_identifier_column(&h)
                && !selected.iter().any(|x| x == &h)
            {
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
            if name.eq_ignore_ascii_case("date") || looks_like_identifier_column(name) {
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
    let s = v.replace([',', '$'], "");
    s.parse::<f64>().ok()
}

fn parse_date_like(v: &str) -> Option<chrono::NaiveDate> {
    let s = v.trim();
    if s.is_empty() {
        return None;
    }
    if s.len() >= 10 && s.as_bytes().get(4) == Some(&b'-') && s.as_bytes().get(7) == Some(&b'-') {
        if let Ok(d) = chrono::NaiveDate::parse_from_str(&s[..10], "%Y-%m-%d") {
            return Some(d);
        }
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
    if let Ok(dt) = chrono::DateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%:z") {
        return Some(dt.date_naive());
    }
    if let Ok(dt) = chrono::DateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f%:z") {
        return Some(dt.date_naive());
    }
    if let Ok(dt) = chrono::DateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%z") {
        return Some(dt.date_naive());
    }
    if let Ok(dt) = chrono::DateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f%z") {
        return Some(dt.date_naive());
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
        let rows = rdr.records().map(|r| r.expect("row")).collect::<Vec<_>>();
        (headers, rows)
    }

    fn investigate_workflow_args() -> InvestigateWorkflowArgs {
        InvestigateWorkflowArgs {
            question: "why did revenue change".to_string(),
            mode: None,
            config: None,
            profile: None,
            profile_config: None,
            base: Some(PathBuf::from("artifacts/base.json")),
            new: Some(PathBuf::from("artifacts/new.json")),
            postgres_url: None,
            postgres_ssl_mode: PostgresSslMode::Prefer,
            postgres_ca_file: None,
            query: None,
            query_file: None,
            metric: None,
            date_column: None,
            time_grain: None,
            period: None,
            anchor_date: None,
            current_start: None,
            current_end: None,
            previous_start: None,
            previous_end: None,
            dimensions: vec!["region".to_string()],
            drill_fields: vec!["channel".to_string()],
            max_depth: 2,
            max_branches: 2,
            min_contribution: 5.0,
            min_delta_abs: 0.0,
            min_score_improvement: 0.0,
            min_slice_rows: 5,
            top_movers: 2,
            planner: InvestigationPlanner::Deterministic,
            planner_backend: BackendArg::Local,
            planner_model: None,
            verbose: false,
            trace: false,
            output_format: InvestigateOutputFormat::Json,
            out: None,
        }
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
    fn apply_driver_filters_matches_column_name_and_expression() {
        let included = apply_driver_filters(
            vec![
                DriverSpec {
                    label: "sum(revenue_usd)".to_string(),
                    col_idx: Some(1),
                    agg: DriverAgg::Sum,
                },
                DriverSpec {
                    label: "count_distinct(channel)".to_string(),
                    col_idx: Some(2),
                    agg: DriverAgg::CountDistinct,
                },
            ],
            &["revenue_usd".to_string()],
            &[],
        );
        assert_eq!(included.len(), 1);
        assert_eq!(included[0].label, "sum(revenue_usd)");

        let excluded = apply_driver_filters(
            vec![
                DriverSpec {
                    label: "sum(revenue_usd)".to_string(),
                    col_idx: Some(1),
                    agg: DriverAgg::Sum,
                },
                DriverSpec {
                    label: "count_distinct(channel)".to_string(),
                    col_idx: Some(2),
                    agg: DriverAgg::CountDistinct,
                },
            ],
            &[],
            &["channel".to_string()],
        );
        assert_eq!(excluded.len(), 1);
        assert_eq!(excluded[0].label, "sum(revenue_usd)");
    }

    #[test]
    fn auto_select_numeric_drivers_skips_count_like_columns() {
        let headers = StringRecord::from(vec![
            "created_at",
            "revenue_usd",
            "net_margin_usd",
            "event_count_per_group",
            "plan_flag",
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
        let names = selected
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();

        assert!(names.contains(&"net_margin_usd".to_string()));
        assert!(!names.contains(&"event_count_per_group".to_string()));
        assert!(!names.contains(&"plan_flag".to_string()));
    }

    #[test]
    fn auto_select_numeric_drivers_respects_zero_top_n() {
        let headers = StringRecord::from(vec!["date", "revenue_usd", "units"]);
        let rows = (0..40)
            .map(|i| {
                StringRecord::from(vec![
                    format!("2026-03-{:02}", (i % 28) + 1),
                    format!("{}", 100.0 + (i as f64)),
                    format!("{}", 10 + i),
                ])
            })
            .collect::<Vec<_>>();

        let selected = auto_select_numeric_drivers(&headers, &rows, 1, 0, 0, None);
        assert!(selected.is_empty());
    }

    #[test]
    fn auto_select_numeric_drivers_handles_sparse_metric_values() {
        let headers = StringRecord::from(vec!["date", "revenue_usd", "units"]);
        let rows = (0..120)
            .map(|i| {
                let metric = if i < 24 {
                    format!("{}", 100.0 + (i as f64))
                } else {
                    "".to_string()
                };
                StringRecord::from(vec![
                    format!("2026-03-{:02}", (i % 28) + 1),
                    metric,
                    format!("{}", 1000 + i),
                ])
            })
            .collect::<Vec<_>>();

        let selected = auto_select_numeric_drivers(&headers, &rows, 1, 0, 3, None);
        let names = selected
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"units".to_string()));
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
            driver_include: vec![],
            driver_exclude: vec![],
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
    fn default_suggest_profile_path_uses_selected_format() {
        let base = Path::new("artifacts/demo_suggest.md");
        assert_eq!(
            default_suggest_profile_path(base, SuggestProfileFormat::Toml),
            PathBuf::from("artifacts/demo_suggest.toml")
        );
        assert_eq!(
            default_suggest_profile_path(base, SuggestProfileFormat::Json),
            PathBuf::from("artifacts/demo_suggest.json")
        );
    }

    #[test]
    fn build_suggested_profile_config_supports_json() {
        let report = AnalyzeSuggestReport {
            input: "data/demo.csv".to_string(),
            sampled_rows: 10,
            sample_mode: "head".to_string(),
            sample_seed: 42,
            profile_name: "demo".to_string(),
            suggested_group_by: vec!["region".to_string()],
            suggested_metrics: vec!["revenue_usd".to_string()],
            suggested_rank_by: Some("revenue_usd".to_string()),
            suggested_date_column: Some("order_date".to_string()),
            warnings: vec![],
            columns: vec![],
        };
        let raw = build_suggested_profile_config(&report, "demo", 3, 3, SuggestProfileFormat::Json);
        let v: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        assert_eq!(v["profiles"]["demo"]["group_by"][0], "region");
        assert_eq!(v["profiles"]["demo"]["metrics"][0], "revenue_usd");
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
        assert!(
            (actual - predicted).abs() < 1e-6,
            "actual={} predicted={}",
            actual,
            predicted
        );
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
            &headers, &rows, &rows, metric_idx, &identity, 0.1, 5_970.47,
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
            &headers, &rows, &rows, metric_idx, &identity, 1.5, 77_765.73,
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

    #[test]
    fn route_investigation_mode_uses_expected_keywords() {
        assert_eq!(
            route_investigation_mode("Why did concentration increase last month?"),
            InvestigationMode::ConcentrationDrivers
        );
        assert_eq!(
            route_investigation_mode("Compare current vs baseline and list biggest shifts"),
            InvestigationMode::CompareSnapshots
        );
        assert_eq!(
            route_investigation_mode("What should I inspect next?"),
            InvestigationMode::RecommendNext
        );
        assert_eq!(
            route_investigation_mode("Why did revenue change?"),
            InvestigationMode::ChangeDrivers
        );
    }

    #[test]
    fn resolve_investigation_mode_prefers_explicit_mode_over_question() {
        let mut args = investigate_workflow_args();
        args.question = "What should I inspect next?".to_string();
        args.mode = Some(InvestigationModeArg::ChangeDrivers);
        assert_eq!(
            resolve_investigation_mode(&args),
            InvestigationMode::ChangeDrivers
        );
    }

    #[test]
    fn investigate_config_applies_defaults_when_cli_not_set() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let cfg_path = std::env::temp_dir().join(format!("factorlens_investigate_cfg_{}.toml", ts));
        fs::write(
            &cfg_path,
            r#"
[investigate]
question = "Why did config question apply?"
mode = "compare_snapshots"
metric = "revenue_usd"
dimensions = ["region", "channel"]
drill_fields = ["channel"]
max_depth = 5
planner = "llm"
planner_backend = "local"
planner_model = "/models/llama.gguf"
"#,
        )
        .expect("write config");

        let mut args = investigate_workflow_args();
        args.config = Some(cfg_path.clone());
        args.question = String::new();
        args.metric = None;
        args.dimensions = vec![];
        args.drill_fields = vec![];
        args.max_depth = 2;
        args.planner = InvestigationPlanner::Deterministic;
        args.planner_backend = BackendArg::Local;
        args.planner_model = None;

        let out = apply_investigate_config(args).expect("apply config");
        assert_eq!(out.question, "Why did config question apply?");
        assert_eq!(out.mode, Some(InvestigationModeArg::CompareSnapshots));
        assert_eq!(out.metric.as_deref(), Some("revenue_usd"));
        assert_eq!(
            out.dimensions,
            vec!["region".to_string(), "channel".to_string()]
        );
        assert_eq!(out.drill_fields, vec!["channel".to_string()]);
        assert_eq!(out.max_depth, 5);
        assert_eq!(out.planner, InvestigationPlanner::Llm);
        assert!(matches!(out.planner_backend, BackendArg::Local));
        assert_eq!(out.planner_model.as_deref(), Some("/models/llama.gguf"));

        let _ = fs::remove_file(&cfg_path);
    }

    #[test]
    fn investigate_profile_config_applies_named_profile() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let cfg_path =
            std::env::temp_dir().join(format!("factorlens_investigate_profiles_{}.toml", ts));
        fs::write(
            &cfg_path,
            r#"
[profiles.change]
question = "Why did revenue change?"
mode = "change_drivers"
metric = "revenue_usd"
dimensions = ["region", "channel"]
"#,
        )
        .expect("write config");

        let mut args = investigate_workflow_args();
        args.question = String::new();
        args.metric = None;
        args.dimensions = vec![];
        args.profile = Some("change".to_string());
        args.profile_config = Some(cfg_path.clone());

        let out = apply_investigate_config(args).expect("apply config");
        assert_eq!(out.question, "Why did revenue change?");
        assert_eq!(out.mode, Some(InvestigationModeArg::ChangeDrivers));
        assert_eq!(out.metric.as_deref(), Some("revenue_usd"));
        assert_eq!(
            out.dimensions,
            vec!["region".to_string(), "channel".to_string()]
        );

        let _ = fs::remove_file(&cfg_path);
    }

    #[test]
    fn investigate_config_with_multiple_profiles_requires_profile_name() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let cfg_path =
            std::env::temp_dir().join(format!("factorlens_investigate_profiles_multi_{}.toml", ts));
        fs::write(
            &cfg_path,
            r#"
[profiles.change]
question = "Why did revenue change?"

[profiles.conc]
question = "Why did concentration increase?"
"#,
        )
        .expect("write config");

        let mut args = investigate_workflow_args();
        args.config = Some(cfg_path.clone());
        let err = apply_investigate_config(args)
            .err()
            .expect("expected multi-profile error");
        assert!(err.to_string().contains("multiple [profiles.*] entries"));

        let _ = fs::remove_file(&cfg_path);
    }

    #[test]
    fn investigate_profile_config_requires_profile_name() {
        let mut args = investigate_workflow_args();
        args.profile = None;
        args.profile_config = Some(PathBuf::from("profiles/investigate.example.toml"));
        let err = apply_investigate_config(args)
            .err()
            .expect("expected profile_config validation error");
        assert!(err.to_string().contains("requires --profile"));
    }

    #[test]
    fn investigate_config_does_not_override_non_default_cli_values() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let cfg_path =
            std::env::temp_dir().join(format!("factorlens_investigate_cfg_override_{}.toml", ts));
        fs::write(
            &cfg_path,
            r#"
[investigate]
max_depth = 3
min_contribution = 2.0
"#,
        )
        .expect("write config");

        let mut args = investigate_workflow_args();
        args.config = Some(cfg_path.clone());
        args.max_depth = 7;
        args.min_contribution = 9.0;
        let out = apply_investigate_config(args).expect("apply config");
        assert_eq!(out.max_depth, 7);
        assert_eq!(out.min_contribution, 9.0);

        let _ = fs::remove_file(&cfg_path);
    }

    #[test]
    fn investigate_config_dedups_dimensions_and_drill_fields() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let cfg_path =
            std::env::temp_dir().join(format!("factorlens_investigate_cfg_dedup_{}.toml", ts));
        fs::write(
            &cfg_path,
            r#"
[investigate]
dimensions = ["region", "Region", "channel", "region"]
drill_fields = ["channel", "channel", "product_line", "CHANNEL"]
"#,
        )
        .expect("write config");

        let mut args = investigate_workflow_args();
        args.config = Some(cfg_path.clone());
        args.dimensions = vec![];
        args.drill_fields = vec![];
        let out = apply_investigate_config(args).expect("apply config");
        assert_eq!(
            out.dimensions,
            vec!["region".to_string(), "channel".to_string()]
        );
        assert_eq!(
            out.drill_fields,
            vec!["channel".to_string(), "product_line".to_string()]
        );

        let _ = fs::remove_file(&cfg_path);
    }

    #[test]
    fn parse_date_like_supports_postgres_timestamptz_variants() {
        let expected = chrono::NaiveDate::from_ymd_opt(2026, 3, 12).expect("valid date");
        assert_eq!(parse_date_like("2026-03-12 14:23:55+00"), Some(expected));
        assert_eq!(
            parse_date_like("2026-03-12 14:23:55.123456+00"),
            Some(expected)
        );
        assert_eq!(parse_date_like("2026-03-12 14:23:55+0000"), Some(expected));
    }

    #[test]
    fn detect_investigate_input_kind_rejects_mixed_extensions() {
        let err = detect_investigate_input_kind(
            Path::new("artifacts/base.json"),
            Path::new("data/new.csv"),
        )
        .expect_err("expected extension mismatch error");
        assert!(
            err.to_string()
                .contains("both be JSON artifacts or both be CSV datasets"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn select_drill_candidates_respect_thresholds_and_branch_limit() {
        let step = InvestigationStep {
            depth: 0,
            dimension: "region".to_string(),
            scope: vec![],
            primary_metric: "revenue_usd".to_string(),
            base_records: 100,
            new_records: 100,
            segment_count: 2,
            top5_concentration_base_pct: 60.0,
            top5_concentration_new_pct: 67.0,
            top5_concentration_delta_pp: 7.0,
            top1_concentration_base_pct: 60.0,
            top1_concentration_new_pct: 67.0,
            top1_concentration_delta_pp: 7.0,
            movers: vec![
                InvestigationMover {
                    segment: "West".to_string(),
                    base_records: 60,
                    new_records: 64,
                    base_share_pct: 60.0,
                    new_share_pct: 64.0,
                    delta_share_pp: 4.0,
                    base_primary_metric_value: 1_000.0,
                    new_primary_metric_value: 1_080.0,
                    delta_primary_metric_value: 80.0,
                },
                InvestigationMover {
                    segment: "(blank)".to_string(),
                    base_records: 20,
                    new_records: 20,
                    base_share_pct: 20.0,
                    new_share_pct: 20.0,
                    delta_share_pp: 0.0,
                    base_primary_metric_value: 300.0,
                    new_primary_metric_value: 300.0,
                    delta_primary_metric_value: 0.0,
                },
            ],
        };

        let picked =
            select_drill_candidates(&step, InvestigationMode::ChangeDrivers, 50.0, 0.0, 10, 2);
        assert_eq!(picked.len(), 1, "expected one eligible mover");
        assert_eq!(picked[0].segment, "West");

        let none =
            select_drill_candidates(&step, InvestigationMode::ChangeDrivers, 120.0, 0.0, 10, 2);
        assert!(none.is_empty(), "expected threshold to block selection");

        let blocked_by_delta =
            select_drill_candidates(&step, InvestigationMode::ChangeDrivers, 50.0, 100.0, 10, 2);
        assert!(
            blocked_by_delta.is_empty(),
            "expected min_delta_abs threshold to block selection"
        );

        let concentration_ignores_delta_abs = select_drill_candidates(
            &step,
            InvestigationMode::ConcentrationDrivers,
            1.0,
            100.0,
            10,
            2,
        );
        assert_eq!(
            concentration_ignores_delta_abs.len(),
            1,
            "concentration mode should not filter by min_delta_abs"
        );
    }

    #[test]
    fn build_branch_summary_rows_uses_dimension_scoped_root_scores() {
        let steps = vec![
            InvestigationStep {
                depth: 0,
                dimension: "region".to_string(),
                scope: vec![],
                primary_metric: "revenue_usd".to_string(),
                base_records: 100,
                new_records: 100,
                segment_count: 2,
                top5_concentration_base_pct: 100.0,
                top5_concentration_new_pct: 100.0,
                top5_concentration_delta_pp: 0.0,
                top1_concentration_base_pct: 60.0,
                top1_concentration_new_pct: 62.0,
                top1_concentration_delta_pp: 2.0,
                movers: vec![
                    InvestigationMover {
                        segment: "US".to_string(),
                        base_records: 60,
                        new_records: 62,
                        base_share_pct: 60.0,
                        new_share_pct: 62.0,
                        delta_share_pp: 2.0,
                        base_primary_metric_value: 1_000.0,
                        new_primary_metric_value: 1_200.0,
                        delta_primary_metric_value: 200.0,
                    },
                    InvestigationMover {
                        segment: "EU".to_string(),
                        base_records: 40,
                        new_records: 38,
                        base_share_pct: 40.0,
                        new_share_pct: 38.0,
                        delta_share_pp: -2.0,
                        base_primary_metric_value: 900.0,
                        new_primary_metric_value: 850.0,
                        delta_primary_metric_value: -50.0,
                    },
                ],
            },
            InvestigationStep {
                depth: 1,
                dimension: "discipline".to_string(),
                scope: vec![],
                primary_metric: "revenue_usd".to_string(),
                base_records: 100,
                new_records: 100,
                segment_count: 2,
                top5_concentration_base_pct: 100.0,
                top5_concentration_new_pct: 100.0,
                top5_concentration_delta_pp: 0.0,
                top1_concentration_base_pct: 55.0,
                top1_concentration_new_pct: 58.0,
                top1_concentration_delta_pp: 3.0,
                movers: vec![
                    InvestigationMover {
                        segment: "Medical".to_string(),
                        base_records: 55,
                        new_records: 58,
                        base_share_pct: 55.0,
                        new_share_pct: 58.0,
                        delta_share_pp: 3.0,
                        base_primary_metric_value: 1_100.0,
                        new_primary_metric_value: 1_300.0,
                        delta_primary_metric_value: 200.0,
                    },
                    InvestigationMover {
                        segment: "Bio".to_string(),
                        base_records: 45,
                        new_records: 42,
                        base_share_pct: 45.0,
                        new_share_pct: 42.0,
                        delta_share_pp: -3.0,
                        base_primary_metric_value: 800.0,
                        new_primary_metric_value: 700.0,
                        delta_primary_metric_value: -100.0,
                    },
                ],
            },
            InvestigationStep {
                depth: 2,
                dimension: "category".to_string(),
                scope: vec![("discipline".to_string(), "Medical".to_string())],
                primary_metric: "revenue_usd".to_string(),
                base_records: 58,
                new_records: 58,
                segment_count: 2,
                top5_concentration_base_pct: 100.0,
                top5_concentration_new_pct: 100.0,
                top5_concentration_delta_pp: 0.0,
                top1_concentration_base_pct: 70.0,
                top1_concentration_new_pct: 72.0,
                top1_concentration_delta_pp: 2.0,
                movers: vec![InvestigationMover {
                    segment: "Oncology".to_string(),
                    base_records: 40,
                    new_records: 42,
                    base_share_pct: 70.0,
                    new_share_pct: 72.0,
                    delta_share_pp: 2.0,
                    base_primary_metric_value: 700.0,
                    new_primary_metric_value: 880.0,
                    delta_primary_metric_value: 180.0,
                }],
            },
        ];

        let rows = build_branch_summary_rows(&steps, InvestigationMode::ChangeDrivers, 3);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].root_dimension, "discipline");
        assert_eq!(rows[0].root_segment, "Medical");
        assert_eq!(rows[0].root_score, 200.0);
    }

    #[test]
    fn resolve_investigation_dimensions_rejects_json_dimension_mismatch() {
        let mut args = investigate_workflow_args();
        args.dimensions = vec!["channel".to_string()];
        let base = serde_json::json!({
            "group_by": ["region"]
        });
        let new = serde_json::json!({
            "group_by": ["region"]
        });
        let err = resolve_investigation_dimensions(
            &args,
            Path::new("base.json"),
            Path::new("new.json"),
            InvestigateInputKind::JsonArtifacts,
            Some(&base),
            Some(&new),
        )
        .expect_err("expected mismatch to fail");
        assert!(
            err.to_string()
                .contains("--dimensions must match artifact grouping"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn resolve_investigation_dimensions_rejects_artifact_group_mismatch() {
        let mut args = investigate_workflow_args();
        args.dimensions = vec![];
        let base = serde_json::json!({
            "group_by": ["region"]
        });
        let new = serde_json::json!({
            "group_by": ["channel"]
        });
        let err = resolve_investigation_dimensions(
            &args,
            Path::new("base.json"),
            Path::new("new.json"),
            InvestigateInputKind::JsonArtifacts,
            Some(&base),
            Some(&new),
        )
        .expect_err("expected artifact mismatch to fail");
        assert!(
            err.to_string()
                .contains("base/new artifacts use different group_by dimensions"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn resolve_investigation_dimensions_autodetects_csv_dimensions() {
        let mut args = investigate_workflow_args();
        args.base = Some(test_data_path("factorlens_demo_sales_100.csv"));
        args.new = Some(test_data_path("factorlens_demo_sales_150.csv"));
        args.metric = Some("revenue_usd".to_string());
        args.dimensions = vec![];
        args.drill_fields = vec![];

        let base = args.base.clone().expect("base");
        let new = args.new.clone().expect("new");
        let (top, drills) = resolve_investigation_dimensions(
            &args,
            &base,
            &new,
            InvestigateInputKind::CsvDatasets,
            None,
            None,
        )
        .expect("expected csv dimension inference to succeed");
        assert_eq!(top, "region");
        assert_eq!(
            drills,
            vec![
                "channel".to_string(),
                "product_line".to_string(),
                "plan_tier".to_string()
            ]
        );
    }

    #[test]
    fn infer_csv_dimensions_excludes_identifier_columns() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let base_path = std::env::temp_dir().join(format!("factorlens_base_{}.csv", ts));
        let new_path = std::env::temp_dir().join(format!("factorlens_new_{}.csv", ts));

        let csv = "\
region,group_id,revenue_usd\n\
North,G-1,100\n\
North,G-2,200\n\
West,G-3,300\n\
East,G-4,400\n\
Other,G-5,500\n";
        fs::write(&base_path, csv).expect("write base");
        fs::write(&new_path, csv).expect("write new");

        let inferred =
            infer_csv_investigation_dimensions(&base_path, &new_path, Some("revenue_usd"))
                .expect("infer dimensions");
        assert!(
            inferred.iter().any(|d| d == "region"),
            "expected region to be inferred"
        );
        assert!(
            !inferred.iter().any(|d| d == "group_id"),
            "identifier columns should be excluded from inferred dimensions"
        );

        let _ = fs::remove_file(&base_path);
        let _ = fs::remove_file(&new_path);
    }

    #[test]
    fn infer_csv_dimensions_keeps_concentrated_high_cardinality_text_columns() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let base_path = std::env::temp_dir().join(format!("factorlens_hc_keep_base_{}.csv", ts));
        let new_path = std::env::temp_dir().join(format!("factorlens_hc_keep_new_{}.csv", ts));

        let mut base_w = csv::Writer::from_path(&base_path).expect("create base");
        let mut new_w = csv::Writer::from_path(&new_path).expect("create new");
        base_w
            .write_record(["organization_name", "region", "revenue_usd"])
            .expect("base header");
        new_w
            .write_record(["organization_name", "region", "revenue_usd"])
            .expect("new header");

        for i in 0..100usize {
            let (org, revenue) = if i < 40 {
                ("MajorOrg".to_string(), "1000".to_string())
            } else {
                (format!("Org_{}", i), "10".to_string())
            };
            base_w
                .write_record([org, "US".to_string(), revenue])
                .expect("base row");
        }
        for i in 0..100usize {
            let (org, revenue) = if i < 45 {
                ("MajorOrg".to_string(), "1200".to_string())
            } else {
                (format!("Org_{}", i + 100), "12".to_string())
            };
            new_w
                .write_record([org, "US".to_string(), revenue])
                .expect("new row");
        }
        base_w.flush().expect("flush base");
        new_w.flush().expect("flush new");

        let inferred =
            infer_csv_investigation_dimensions(&base_path, &new_path, Some("revenue_usd"))
                .expect("infer dimensions");
        assert!(
            inferred.iter().any(|d| d == "organization_name"),
            "expected concentrated high-cardinality dimension to be kept"
        );

        let _ = fs::remove_file(&base_path);
        let _ = fs::remove_file(&new_path);
    }

    #[test]
    fn infer_csv_dimensions_skips_diffuse_high_cardinality_text_columns() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let base_path = std::env::temp_dir().join(format!("factorlens_hc_skip_base_{}.csv", ts));
        let new_path = std::env::temp_dir().join(format!("factorlens_hc_skip_new_{}.csv", ts));

        let mut base_w = csv::Writer::from_path(&base_path).expect("create base");
        let mut new_w = csv::Writer::from_path(&new_path).expect("create new");
        base_w
            .write_record(["organization_name", "region", "revenue_usd"])
            .expect("base header");
        new_w
            .write_record(["organization_name", "region", "revenue_usd"])
            .expect("new header");

        for i in 0..120usize {
            base_w
                .write_record([format!("Org_{}", i), "US".to_string(), "100".to_string()])
                .expect("base row");
            new_w
                .write_record([
                    format!("Org_{}", i + 200),
                    "US".to_string(),
                    "100".to_string(),
                ])
                .expect("new row");
        }
        base_w.flush().expect("flush base");
        new_w.flush().expect("flush new");

        let inferred =
            infer_csv_investigation_dimensions(&base_path, &new_path, Some("revenue_usd"))
                .expect("infer dimensions");
        assert!(
            !inferred.iter().any(|d| d == "organization_name"),
            "expected diffuse high-cardinality dimension to be skipped"
        );

        let _ = fs::remove_file(&base_path);
        let _ = fs::remove_file(&new_path);
    }

    #[test]
    fn infer_column_role_excludes_identifier_like_dimension_candidates() {
        let role = infer_column_role("group_id", 95.0, 35, 0.05, 0.0);
        assert_ne!(
            role, "dimension",
            "identifier-like columns should not be inferred as dimensions"
        );
    }

    #[test]
    fn auto_detect_groups_excludes_identifier_columns() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let csv_path = std::env::temp_dir().join(format!("factorlens_groups_{}.csv", ts));
        let csv = "\
region,group_id,category\n\
North,G-1,A\n\
North,G-2,A\n\
West,G-3,B\n\
West,G-4,B\n\
East,G-5,C\n";
        fs::write(&csv_path, csv).expect("write csv");
        let headers = StringRecord::from(vec!["region", "group_id", "category"]);
        let groups = auto_detect_groups(&headers, &csv_path, 3).expect("auto groups");
        assert!(groups.iter().any(|g| g == "region") || groups.iter().any(|g| g == "category"));
        assert!(
            !groups.iter().any(|g| g == "group_id"),
            "identifier columns should be excluded from auto group suggestions"
        );
        let _ = fs::remove_file(&csv_path);
    }

    #[test]
    fn investigate_step_json_mode_errors_when_metric_missing() {
        let mut args = investigate_workflow_args();
        args.metric = Some("revenue_usd".to_string());
        let base = serde_json::json!({
            "records": 10,
            "top5_count": 5,
            "primary_metric": "profit_usd",
            "metrics": ["profit_usd"],
            "group_by": ["region"],
            "groups": [{"group":"US", "count":5, "count_share_pct":50.0, "profit_usd":100.0}]
        });
        let new = serde_json::json!({
            "records": 10,
            "top5_count": 5,
            "primary_metric": "profit_usd",
            "metrics": ["profit_usd"],
            "group_by": ["region"],
            "groups": [{"group":"US", "count":5, "count_share_pct":50.0, "profit_usd":130.0}]
        });

        let input = InvestigationInputRefs {
            base_path: Path::new("base.json"),
            new_path: Path::new("new.json"),
            input_kind: InvestigateInputKind::JsonArtifacts,
            base_artifact: Some(&base),
            new_artifact: Some(&new),
        };
        let err = investigation_step_from_inputs(
            &args,
            &input,
            "region",
            &[],
            InvestigationMode::ChangeDrivers,
        )
        .expect_err("expected missing metric to fail");
        assert!(
            err.to_string()
                .contains("does not contain metric 'revenue_usd'"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn investigate_step_json_mode_errors_when_primary_metric_mismatch_without_metric() {
        let mut args = investigate_workflow_args();
        args.metric = None;
        let base = serde_json::json!({
            "records": 10,
            "top5_count": 5,
            "primary_metric": "revenue_usd",
            "metrics": ["revenue_usd"],
            "group_by": ["region"],
            "groups": [{"group":"US", "count":5, "count_share_pct":50.0, "revenue_usd":100.0}]
        });
        let new = serde_json::json!({
            "records": 10,
            "top5_count": 5,
            "primary_metric": "profit_usd",
            "metrics": ["profit_usd"],
            "group_by": ["region"],
            "groups": [{"group":"US", "count":5, "count_share_pct":50.0, "profit_usd":130.0}]
        });

        let input = InvestigationInputRefs {
            base_path: Path::new("base.json"),
            new_path: Path::new("new.json"),
            input_kind: InvestigateInputKind::JsonArtifacts,
            base_artifact: Some(&base),
            new_artifact: Some(&new),
        };
        let err = investigation_step_from_inputs(
            &args,
            &input,
            "region",
            &[],
            InvestigationMode::ChangeDrivers,
        )
        .expect_err("expected primary metric mismatch to fail");
        assert!(
            err.to_string()
                .contains("base/new artifacts have different primary_metric values"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn investigation_step_uses_max_branches_for_candidate_capacity() {
        let mut args = investigate_workflow_args();
        args.top_movers = 1;
        args.max_branches = 3;
        let base = serde_json::json!({
            "records": 100,
            "top5_count": 50,
            "primary_metric": "revenue_usd",
            "metrics": ["revenue_usd"],
            "group_by": ["region"],
            "groups": [
                {"group":"A", "count":34, "count_share_pct":34.0, "revenue_usd":100.0},
                {"group":"B", "count":33, "count_share_pct":33.0, "revenue_usd":90.0},
                {"group":"C", "count":33, "count_share_pct":33.0, "revenue_usd":80.0}
            ]
        });
        let new = serde_json::json!({
            "records": 100,
            "top5_count": 60,
            "primary_metric": "revenue_usd",
            "metrics": ["revenue_usd"],
            "group_by": ["region"],
            "groups": [
                {"group":"A", "count":30, "count_share_pct":30.0, "revenue_usd":210.0},
                {"group":"B", "count":30, "count_share_pct":30.0, "revenue_usd":180.0},
                {"group":"C", "count":40, "count_share_pct":40.0, "revenue_usd":120.0}
            ]
        });

        let input = InvestigationInputRefs {
            base_path: Path::new("base.json"),
            new_path: Path::new("new.json"),
            input_kind: InvestigateInputKind::JsonArtifacts,
            base_artifact: Some(&base),
            new_artifact: Some(&new),
        };
        let step = investigation_step_from_inputs(
            &args,
            &input,
            "region",
            &[],
            InvestigationMode::ChangeDrivers,
        )
        .expect("step");
        assert_eq!(
            step.movers.len(),
            3,
            "movers should cover max_branches candidates even when top_movers is lower"
        );
    }

    #[test]
    fn metric_matches_expected_accepts_common_aliases() {
        assert!(metric_matches_expected("revenue_usd", "revenue_usd"));
        assert!(metric_matches_expected("Revenue_USD", "revenue_usd"));
        assert!(metric_matches_expected("revenue", "revenue_usd"));
        assert!(metric_matches_expected("conversion", "conversion_rate"));
        assert!(!metric_matches_expected("cost", "revenue_usd"));
    }

    #[test]
    fn validate_llm_planner_action_accepts_metric_alias_and_normalizes_dimension() {
        let mut args = investigate_workflow_args();
        args.metric = Some("revenue_usd".to_string());
        let proposed = LlmPlannerAction {
            action: "analyze_compare".to_string(),
            reason: Some("test".to_string()),
            params: Some(LlmPlannerParams {
                metric: Some("revenue".to_string()),
                group_by: Some(vec!["REGION".to_string()]),
                filters: Some(HashMap::new()),
            }),
        };
        let out = validate_llm_planner_action(
            &proposed,
            &args,
            &["region".to_string(), "channel".to_string()],
            &[],
            InvestigateInputKind::CsvDatasets,
        )
        .expect("alias metric should pass");
        match out {
            InvestigationExecAction::AnalyzeCompare { group_by, .. } => {
                assert_eq!(group_by, "region")
            }
            _ => panic!("expected analyze_compare"),
        }
    }

    #[test]
    fn validate_llm_planner_action_ignores_mismatched_metric_override() {
        let mut args = investigate_workflow_args();
        args.metric = Some("revenue_usd".to_string());
        let proposed = LlmPlannerAction {
            action: "analyze_compare".to_string(),
            reason: Some("test".to_string()),
            params: Some(LlmPlannerParams {
                metric: Some("revenue_concentration".to_string()),
                group_by: Some(vec!["region".to_string()]),
                filters: Some(HashMap::new()),
            }),
        };
        let out = validate_llm_planner_action(
            &proposed,
            &args,
            &["region".to_string(), "channel".to_string()],
            &[],
            InvestigateInputKind::CsvDatasets,
        )
        .expect("mismatched metric should be ignored");
        match out {
            InvestigationExecAction::AnalyzeCompare { group_by, .. } => {
                assert_eq!(group_by, "region")
            }
            _ => panic!("expected analyze_compare"),
        }
    }

    #[test]
    fn validate_llm_planner_action_infers_group_by_when_missing() {
        let mut args = investigate_workflow_args();
        args.metric = Some("revenue_usd".to_string());

        let top_level = LlmPlannerAction {
            action: "analyze_compare".to_string(),
            reason: Some("test".to_string()),
            params: Some(LlmPlannerParams {
                metric: Some("revenue_usd".to_string()),
                group_by: None,
                filters: Some(HashMap::new()),
            }),
        };
        let out0 = validate_llm_planner_action(
            &top_level,
            &args,
            &[
                "region".to_string(),
                "channel".to_string(),
                "product_line".to_string(),
            ],
            &[],
            InvestigateInputKind::CsvDatasets,
        )
        .expect("top-level should infer region");
        match out0 {
            InvestigationExecAction::AnalyzeCompare { group_by, .. } => {
                assert_eq!(group_by, "region")
            }
            _ => panic!("expected analyze_compare"),
        }

        let steps = vec![InvestigationStep {
            depth: 0,
            dimension: "region".to_string(),
            scope: vec![],
            primary_metric: "revenue_usd".to_string(),
            base_records: 100,
            new_records: 150,
            segment_count: 4,
            top5_concentration_base_pct: 100.0,
            top5_concentration_new_pct: 100.0,
            top5_concentration_delta_pp: 0.0,
            top1_concentration_base_pct: 29.0,
            top1_concentration_new_pct: 47.33,
            top1_concentration_delta_pp: 18.33,
            movers: vec![],
        }];
        let drill = LlmPlannerAction {
            action: "drill_down".to_string(),
            reason: Some("test".to_string()),
            params: Some(LlmPlannerParams {
                metric: Some("revenue_usd".to_string()),
                group_by: None,
                filters: Some(HashMap::from([("region".to_string(), "US".to_string())])),
            }),
        };
        let out1 = validate_llm_planner_action(
            &drill,
            &args,
            &[
                "region".to_string(),
                "channel".to_string(),
                "product_line".to_string(),
            ],
            &steps,
            InvestigateInputKind::CsvDatasets,
        )
        .expect("drill should infer next dimension");
        match out1 {
            InvestigationExecAction::DrillDown { group_by, .. } => assert_eq!(group_by, "channel"),
            _ => panic!("expected drill_down"),
        }
    }

    #[test]
    fn validate_llm_planner_action_rejects_first_step_drill_down() {
        let mut args = investigate_workflow_args();
        args.metric = Some("revenue_usd".to_string());
        let proposed = LlmPlannerAction {
            action: "drill_down".to_string(),
            reason: Some("test".to_string()),
            params: Some(LlmPlannerParams {
                metric: Some("revenue_usd".to_string()),
                group_by: Some(vec!["region".to_string()]),
                filters: Some(HashMap::from([("region".to_string(), "US".to_string())])),
            }),
        };
        let err = match validate_llm_planner_action(
            &proposed,
            &args,
            &["region".to_string(), "channel".to_string()],
            &[],
            InvestigateInputKind::CsvDatasets,
        ) {
            Ok(_) => panic!("first step drill_down should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("first llm action must be analyze_compare"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn validate_llm_planner_action_stop_autofollows_drilldown_when_possible() {
        let mut args = investigate_workflow_args();
        args.metric = Some("revenue_usd".to_string());
        args.max_depth = 3;

        let steps = vec![InvestigationStep {
            depth: 0,
            dimension: "region".to_string(),
            scope: vec![],
            primary_metric: "revenue_usd".to_string(),
            base_records: 100,
            new_records: 150,
            segment_count: 4,
            top5_concentration_base_pct: 100.0,
            top5_concentration_new_pct: 100.0,
            top5_concentration_delta_pp: 0.0,
            top1_concentration_base_pct: 29.0,
            top1_concentration_new_pct: 47.33,
            top1_concentration_delta_pp: 18.33,
            movers: vec![InvestigationMover {
                segment: "US".to_string(),
                base_records: 29,
                new_records: 71,
                base_share_pct: 29.0,
                new_share_pct: 47.33,
                delta_share_pp: 18.33,
                base_primary_metric_value: 1.0,
                new_primary_metric_value: 2.0,
                delta_primary_metric_value: 1.0,
            }],
        }];

        let proposed = LlmPlannerAction {
            action: "stop".to_string(),
            reason: Some("previous result indicates a significant change".to_string()),
            params: None,
        };
        let out = validate_llm_planner_action(
            &proposed,
            &args,
            &[
                "region".to_string(),
                "channel".to_string(),
                "product_line".to_string(),
            ],
            &steps,
            InvestigateInputKind::CsvDatasets,
        )
        .expect("stop should auto-follow into one drill-down");
        match out {
            InvestigationExecAction::DrillDown {
                group_by,
                scope,
                reason,
            } => {
                assert_eq!(group_by, "channel");
                assert_eq!(scope, vec![("region".to_string(), "US".to_string())]);
                assert!(
                    reason.contains("auto-follow"),
                    "unexpected reason: {}",
                    reason
                );
            }
            _ => panic!("expected auto drill_down"),
        }
    }

    #[test]
    fn validate_llm_planner_action_auto_pivots_repeated_analyze_to_drilldown() {
        let mut args = investigate_workflow_args();
        args.metric = Some("revenue_usd".to_string());

        let steps = vec![InvestigationStep {
            depth: 0,
            dimension: "region".to_string(),
            scope: vec![],
            primary_metric: "revenue_usd".to_string(),
            base_records: 100,
            new_records: 150,
            segment_count: 4,
            top5_concentration_base_pct: 100.0,
            top5_concentration_new_pct: 100.0,
            top5_concentration_delta_pp: 0.0,
            top1_concentration_base_pct: 29.0,
            top1_concentration_new_pct: 47.33,
            top1_concentration_delta_pp: 18.33,
            movers: vec![InvestigationMover {
                segment: "US".to_string(),
                base_records: 29,
                new_records: 71,
                base_share_pct: 29.0,
                new_share_pct: 47.33,
                delta_share_pp: 18.33,
                base_primary_metric_value: 1.0,
                new_primary_metric_value: 2.0,
                delta_primary_metric_value: 1.0,
            }],
        }];

        let proposed = LlmPlannerAction {
            action: "analyze_compare".to_string(),
            reason: Some("retry top level".to_string()),
            params: Some(LlmPlannerParams {
                metric: Some("revenue_usd".to_string()),
                group_by: Some(vec!["region".to_string()]),
                filters: Some(HashMap::new()),
            }),
        };

        let out = validate_llm_planner_action(
            &proposed,
            &args,
            &[
                "region".to_string(),
                "channel".to_string(),
                "product_line".to_string(),
            ],
            &steps,
            InvestigateInputKind::CsvDatasets,
        )
        .expect("should auto-pivot to drilldown");
        match out {
            InvestigationExecAction::DrillDown {
                group_by, scope, ..
            } => {
                assert_eq!(group_by, "channel");
                assert_eq!(scope, vec![("region".to_string(), "US".to_string())]);
            }
            _ => panic!("expected drill_down"),
        }
    }

    #[test]
    fn validate_llm_planner_action_avoids_grouping_by_scoped_dimension() {
        let mut args = investigate_workflow_args();
        args.metric = Some("revenue_usd".to_string());
        let steps = vec![InvestigationStep {
            depth: 0,
            dimension: "region".to_string(),
            scope: vec![],
            primary_metric: "revenue_usd".to_string(),
            base_records: 100,
            new_records: 150,
            segment_count: 4,
            top5_concentration_base_pct: 100.0,
            top5_concentration_new_pct: 100.0,
            top5_concentration_delta_pp: 0.0,
            top1_concentration_base_pct: 29.0,
            top1_concentration_new_pct: 47.33,
            top1_concentration_delta_pp: 18.33,
            movers: vec![InvestigationMover {
                segment: "US".to_string(),
                base_records: 29,
                new_records: 71,
                base_share_pct: 29.0,
                new_share_pct: 47.33,
                delta_share_pp: 18.33,
                base_primary_metric_value: 4_020_509.65,
                new_primary_metric_value: 13_561_697.76,
                delta_primary_metric_value: 9_541_188.11,
            }],
        }];

        let proposed = LlmPlannerAction {
            action: "analyze_compare".to_string(),
            reason: Some("test".to_string()),
            params: Some(LlmPlannerParams {
                metric: Some("revenue_usd".to_string()),
                group_by: Some(vec!["region".to_string()]),
                filters: Some(
                    [("region".to_string(), "US".to_string())]
                        .into_iter()
                        .collect(),
                ),
            }),
        };

        let out = validate_llm_planner_action(
            &proposed,
            &args,
            &[
                "region".to_string(),
                "channel".to_string(),
                "product_line".to_string(),
            ],
            &steps,
            InvestigateInputKind::CsvDatasets,
        )
        .expect("should auto-adjust scoped group_by");
        match out {
            InvestigationExecAction::AnalyzeCompare {
                group_by, scope, ..
            } => {
                assert_eq!(scope, vec![("region".to_string(), "US".to_string())]);
                assert_ne!(group_by, "region");
                assert_eq!(group_by, "channel");
            }
            _ => panic!("expected analyze_compare"),
        }
    }

    #[test]
    fn validate_llm_planner_action_preserves_prior_scope_chain() {
        let mut args = investigate_workflow_args();
        args.metric = Some("revenue_usd".to_string());
        let steps = vec![
            InvestigationStep {
                depth: 0,
                dimension: "region".to_string(),
                scope: vec![],
                primary_metric: "revenue_usd".to_string(),
                base_records: 100,
                new_records: 120,
                segment_count: 4,
                top5_concentration_base_pct: 80.0,
                top5_concentration_new_pct: 82.0,
                top5_concentration_delta_pp: 2.0,
                top1_concentration_base_pct: 30.0,
                top1_concentration_new_pct: 34.0,
                top1_concentration_delta_pp: 4.0,
                movers: vec![InvestigationMover {
                    segment: "West".to_string(),
                    base_records: 20,
                    new_records: 35,
                    base_share_pct: 20.0,
                    new_share_pct: 29.0,
                    delta_share_pp: 9.0,
                    base_primary_metric_value: 200.0,
                    new_primary_metric_value: 500.0,
                    delta_primary_metric_value: 300.0,
                }],
            },
            InvestigationStep {
                depth: 1,
                dimension: "channel".to_string(),
                scope: vec![("region".to_string(), "West".to_string())],
                primary_metric: "revenue_usd".to_string(),
                base_records: 50,
                new_records: 60,
                segment_count: 3,
                top5_concentration_base_pct: 100.0,
                top5_concentration_new_pct: 100.0,
                top5_concentration_delta_pp: 0.0,
                top1_concentration_base_pct: 45.0,
                top1_concentration_new_pct: 55.0,
                top1_concentration_delta_pp: 10.0,
                movers: vec![InvestigationMover {
                    segment: "Direct".to_string(),
                    base_records: 30,
                    new_records: 40,
                    base_share_pct: 45.0,
                    new_share_pct: 55.0,
                    delta_share_pp: 10.0,
                    base_primary_metric_value: 300.0,
                    new_primary_metric_value: 700.0,
                    delta_primary_metric_value: 400.0,
                }],
            },
        ];

        let proposed = LlmPlannerAction {
            action: "analyze_compare".to_string(),
            reason: Some("Strongest driver is channel".to_string()),
            params: Some(LlmPlannerParams {
                metric: Some("revenue_usd".to_string()),
                group_by: Some(vec!["region".to_string()]),
                filters: Some(
                    [("channel".to_string(), "Direct".to_string())]
                        .into_iter()
                        .collect(),
                ),
            }),
        };

        let out = validate_llm_planner_action(
            &proposed,
            &args,
            &[
                "region".to_string(),
                "channel".to_string(),
                "product_line".to_string(),
            ],
            &steps,
            InvestigateInputKind::CsvDatasets,
        )
        .expect("should preserve prior scope and choose non-scoped group_by");
        match out {
            InvestigationExecAction::AnalyzeCompare {
                group_by, scope, ..
            } => {
                assert_eq!(
                    scope,
                    vec![
                        ("region".to_string(), "West".to_string()),
                        ("channel".to_string(), "Direct".to_string())
                    ]
                );
                assert_eq!(group_by, "product_line");
            }
            _ => panic!("expected analyze_compare"),
        }
    }

    #[test]
    fn normalize_planner_reason_rewrites_placeholder_text() {
        assert_eq!(
            normalize_planner_reason("drill_down", "previous_result"),
            "planner selected drill-down from prior top mover"
        );
        assert_eq!(
            normalize_planner_reason(
                "analyze_compare",
                "previous results indicate a need for comparison"
            ),
            "planner selected top-level comparison"
        );
    }

    #[test]
    fn parse_llm_planner_action_handles_wrapped_json() {
        let raw = "planner: ok\n{\"action\":\"analyze_compare\",\"reason\":\"top\",\"params\":{\"group_by\":[\"region\"],\"filters\":{}}}\nextra";
        let parsed = parse_llm_planner_action(raw).expect("parse wrapped json");
        assert_eq!(parsed.action, "analyze_compare");
    }

    #[test]
    fn parse_llm_planner_action_prefers_last_json_object() {
        let raw = concat!(
            "{\"action\":\"analyze_compare\",\"reason\":\"first\",\"params\":{\"group_by\":[\"region\"],\"filters\":{}}}\n",
            "{\"action\":\"drill_down\",\"reason\":\"second\",\"params\":{\"group_by\":[\"channel\"],\"filters\":{\"region\":\"US\"}}}\n"
        );
        let parsed = parse_llm_planner_action(raw).expect("parse last json");
        assert_eq!(parsed.action, "drill_down");
    }

    #[test]
    fn sanitize_llm_summary_strips_prompt_and_json_noise() {
        let raw = r#"Provide 2-4 short lines. Keep it concise and grounded.
{
  "question": "Why did revenue concentration increase?"
}
assistant
Revenue concentration increased mainly due to US growth.
Within US, Direct gained share while Marketplace lost share.
Further drill-down stopped at max depth.
"#;
        let cleaned = sanitize_llm_summary(raw, "Why did revenue concentration increase?");
        assert!(cleaned.contains("Revenue concentration increased mainly due to US growth."));
        assert!(!cleaned.contains("\"question\""));
        assert!(!cleaned.contains("Provide 2-4 short lines"));
    }

    #[test]
    fn sanitize_llm_summary_trims_trailing_pipe_noise() {
        let raw = "Revenue concentration increased in US. |\nWithin US, Direct gained share. |";
        let cleaned = sanitize_llm_summary(raw, "Why did revenue concentration increase?");
        assert!(!cleaned.contains('|'));
        assert!(cleaned.contains("Revenue concentration increased in US."));
    }

    #[test]
    fn llm_summary_quality_rejects_long_decimals_and_prompt_tokens() {
        let steps = vec![InvestigationStep {
            depth: 0,
            dimension: "region".to_string(),
            scope: vec![],
            primary_metric: "revenue_usd".to_string(),
            base_records: 100,
            new_records: 150,
            segment_count: 4,
            top5_concentration_base_pct: 100.0,
            top5_concentration_new_pct: 100.0,
            top5_concentration_delta_pp: 0.0,
            top1_concentration_base_pct: 29.0,
            top1_concentration_new_pct: 47.33,
            top1_concentration_delta_pp: 18.33,
            movers: vec![InvestigationMover {
                segment: "US".to_string(),
                base_records: 29,
                new_records: 71,
                base_share_pct: 29.0,
                new_share_pct: 47.33,
                delta_share_pp: 18.33,
                base_primary_metric_value: 4_020_509.65,
                new_primary_metric_value: 13_561_697.76,
                delta_primary_metric_value: 9_541_188.11,
            }],
        }];
        assert!(!llm_summary_is_usable(
            "Revenue moved from 28.999999 to 47.333333 [USER]",
            &steps,
            "reached max depth 2"
        ));
        assert!(llm_summary_is_usable(
            "US share increased from 29.00% to 47.33% due to Direct growth.",
            &steps,
            "reached max depth 2"
        ));
        assert!(!llm_summary_is_usable(
            "US delta was 954118.11 which drove concentration.",
            &steps,
            "reached max depth 2"
        ));
        assert!(!llm_summary_is_usable(
            "The revenue concentration increased because the US region's share",
            &steps,
            "reached max depth 2"
        ));
    }

    #[test]
    fn recommended_next_question_skips_zero_signal_mover() {
        let step = InvestigationStep {
            depth: 1,
            dimension: "region".to_string(),
            scope: vec![("region".to_string(), "US".to_string())],
            primary_metric: "revenue_usd".to_string(),
            base_records: 50,
            new_records: 50,
            segment_count: 1,
            top5_concentration_base_pct: 100.0,
            top5_concentration_new_pct: 100.0,
            top5_concentration_delta_pp: 0.0,
            top1_concentration_base_pct: 100.0,
            top1_concentration_new_pct: 100.0,
            top1_concentration_delta_pp: 0.0,
            movers: vec![InvestigationMover {
                segment: "US".to_string(),
                base_records: 50,
                new_records: 50,
                base_share_pct: 100.0,
                new_share_pct: 100.0,
                delta_share_pp: 0.0,
                base_primary_metric_value: 100.0,
                new_primary_metric_value: 100.0,
                delta_primary_metric_value: 0.0,
            }],
        };

        let q = recommended_next_question(InvestigationMode::ConcentrationDrivers, Some(&step));
        assert!(q.contains("reduce top-5 concentration"));
        assert!(!q.contains("Drill deeper"));
    }

    #[test]
    fn recommended_next_question_includes_scope_and_signed_delta() {
        let step = InvestigationStep {
            depth: 4,
            dimension: "plan_tier".to_string(),
            scope: vec![
                ("region".to_string(), "West".to_string()),
                ("channel".to_string(), "Direct".to_string()),
                ("product_line".to_string(), "Core".to_string()),
            ],
            primary_metric: "revenue_usd".to_string(),
            base_records: 20,
            new_records: 20,
            segment_count: 3,
            top5_concentration_base_pct: 100.0,
            top5_concentration_new_pct: 100.0,
            top5_concentration_delta_pp: 0.0,
            top1_concentration_base_pct: 60.0,
            top1_concentration_new_pct: 20.0,
            top1_concentration_delta_pp: -40.0,
            movers: vec![InvestigationMover {
                segment: "Premium".to_string(),
                base_records: 10,
                new_records: 3,
                base_share_pct: 50.0,
                new_share_pct: 9.17,
                delta_share_pp: -40.83,
                base_primary_metric_value: 800_000.0,
                new_primary_metric_value: 508_576.75,
                delta_primary_metric_value: -291_423.25,
            }],
        };

        let q = recommended_next_question(InvestigationMode::ConcentrationDrivers, Some(&step));
        assert!(q.contains("within scope [region=West"));
        assert!(q.contains("plan_tier='Premium'"));
        assert!(q.contains("delta share -40.83 pp"));
    }

    #[test]
    fn sanitize_explain_analyze_answer_normalizes_to_bullets() {
        let raw =
            "Here is the summary:\n- Revenue grew 10.00% [E1]\n- Top segment changed 5.00 [E2]";
        let out = sanitize_explain_analyze_answer(raw, 3);
        assert!(out.lines().all(|l| l.trim_start().starts_with('-')));
        assert!(out.contains("[E1]"));
    }

    #[test]
    fn sanitize_explain_analyze_answer_strips_inline_markdown_noise() {
        let raw = "- Investigate the **//Other export//** segment [E1]\n- Revenue is `10.00` [E2]";
        let out = sanitize_explain_analyze_answer(raw, 3);
        assert!(
            !out.contains("**"),
            "unexpected markdown emphasis in: {}",
            out
        );
        assert!(
            !out.contains("`"),
            "unexpected markdown code ticks in: {}",
            out
        );
        assert!(!out.contains("//"), "unexpected slash emphasis in: {}", out);
        assert!(
            out.contains("[E1]"),
            "evidence citation should be preserved"
        );
    }

    #[test]
    fn explain_analyze_answer_requires_citations_and_grounded_numbers() {
        let evidence = vec![
            "group='US' records=10 revenue_usd=100.00".to_string(),
            "group='EU' records=5 revenue_usd=50.00".to_string(),
        ];
        assert!(explain_analyze_answer_is_usable(
            "- US revenue is 100.00 [E1]",
            &evidence
        ));
        assert!(!explain_analyze_answer_is_usable(
            "- US revenue is 999.99 [E1]",
            &evidence
        ));
        assert!(!explain_analyze_answer_is_usable(
            "- US revenue is 100.00",
            &evidence
        ));
    }

    #[test]
    fn build_analysis_evidence_supports_investigate_schema() {
        let v = serde_json::json!({
            "question": "Why did revenue change?",
            "mode": "change_drivers",
            "stopping_reason": "reached max depth 3",
            "recommended_next_question": "Drill into provider",
            "major_global_changes": [
                {
                    "dimension": "organization_name",
                    "segment": "Acme",
                    "primary_metric": "revenue_usd",
                    "delta_primary_metric_value": -1200.5,
                    "delta_share_pp": 2.3
                }
            ],
            "steps": [
                {
                    "depth": 0,
                    "dimension": "region",
                    "primary_metric": "revenue_usd",
                    "segment_count": 6,
                    "top1_concentration_base_pct": 35.0,
                    "top1_concentration_new_pct": 42.0,
                    "top1_concentration_delta_pp": 7.0,
                    "top5_concentration_base_pct": 88.0,
                    "top5_concentration_new_pct": 91.0,
                    "top5_concentration_delta_pp": 3.0,
                    "movers": [
                        {
                            "segment": "US",
                            "base_primary_metric_value": 10000.0,
                            "new_primary_metric_value": 12000.0,
                            "delta_primary_metric_value": 2000.0,
                            "delta_share_pp": 5.5
                        }
                    ]
                }
            ]
        });
        let evidence = build_analysis_evidence(&v);
        assert!(
            evidence
                .iter()
                .any(|e| e.contains("schema") || e.contains("investigate mode")),
            "missing investigate summary evidence: {:?}",
            evidence
        );
        assert!(
            evidence
                .iter()
                .any(|e| e.contains("major_change") && e.contains("Acme")),
            "missing major change evidence: {:?}",
            evidence
        );
        assert!(
            evidence
                .iter()
                .any(|e| e.contains("top_level_strongest") && e.contains("US")),
            "missing top-level strongest mover evidence: {:?}",
            evidence
        );
    }

    #[test]
    fn build_analysis_prompt_context_marks_investigate_schema() {
        let v = serde_json::json!({
            "question": "Why did revenue change?",
            "mode": "change_drivers",
            "planner": "deterministic",
            "stopping_reason": "reached max depth 2",
            "steps": []
        });
        let context = build_analysis_prompt_context(&v, &["sample evidence".to_string()]);
        assert!(context.contains("schema=investigate"));
        assert!(context.contains("mode=change_drivers"));
    }
}

fn build_analysis_prompt_context(v: &serde_json::Value, evidence: &[String]) -> String {
    if is_investigate_analysis_json(v) {
        let question = v
            .get("question")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown");
        let mode = v.get("mode").and_then(|x| x.as_str()).unwrap_or("unknown");
        let planner = v
            .get("planner")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown");
        let steps = v
            .get("steps")
            .and_then(|x| x.as_array())
            .map(|xs| xs.len())
            .unwrap_or(0);
        let stop_reason = v
            .get("stopping_reason")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown");
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
        return format!(
            "schema=investigate | question={} | mode={} | planner={} | steps={} | stopping_reason={}\nevidence:\n{}",
            question, mode, planner, steps, stop_reason, evidence_block
        );
    }

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
        .unwrap_or_default();
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
    if is_investigate_analysis_json(v) {
        return build_investigate_analysis_evidence(v);
    }

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

fn is_investigate_analysis_json(v: &serde_json::Value) -> bool {
    v.get("steps").and_then(|x| x.as_array()).is_some()
}

fn investigate_scope_to_text(scope: &serde_json::Value) -> String {
    let Some(items) = scope.as_array() else {
        return "global".to_string();
    };
    let mut parts = Vec::<String>::new();
    for item in items {
        let Some(pair) = item.as_array() else {
            continue;
        };
        if pair.len() < 2 {
            continue;
        }
        let Some(k) = pair.first().and_then(|x| x.as_str()) else {
            continue;
        };
        let Some(v) = pair.get(1).and_then(|x| x.as_str()) else {
            continue;
        };
        parts.push(format!("{}={}", k, v));
    }
    if parts.is_empty() {
        "global".to_string()
    } else {
        parts.join(", ")
    }
}

fn build_investigate_analysis_evidence(v: &serde_json::Value) -> Vec<String> {
    let mut out = Vec::<String>::new();
    let mode = v.get("mode").and_then(|x| x.as_str()).unwrap_or("unknown");
    let question = v
        .get("question")
        .and_then(|x| x.as_str())
        .unwrap_or("unknown");
    out.push(format!(
        "investigate mode='{}' question='{}'",
        mode, question
    ));

    let steps = v
        .get("steps")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    if let Some(step0) = steps.first() {
        let dimension = step0
            .get("dimension")
            .and_then(|x| x.as_str())
            .unwrap_or("(unknown)");
        let metric = step0
            .get("primary_metric")
            .and_then(|x| x.as_str())
            .unwrap_or("primary_metric");
        let segment_count = step0
            .get("segment_count")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let top1_base = step0
            .get("top1_concentration_base_pct")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        let top1_new = step0
            .get("top1_concentration_new_pct")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        let top1_delta = step0
            .get("top1_concentration_delta_pp")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        let top5_base = step0
            .get("top5_concentration_base_pct")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        let top5_new = step0
            .get("top5_concentration_new_pct")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        let top5_delta = step0
            .get("top5_concentration_delta_pp")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);

        let movers = step0
            .get("movers")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        let base_total = movers
            .iter()
            .map(|m| {
                m.get("base_primary_metric_value")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0)
            })
            .sum::<f64>();
        let new_total = movers
            .iter()
            .map(|m| {
                m.get("new_primary_metric_value")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0)
            })
            .sum::<f64>();
        out.push(format!(
            "top_level grouped_by='{}' segments={} {} base_total={} new_total={} delta={} top1_base_pct={:.2} top1_new_pct={:.2} top1_delta_pp={:+.2} top5_base_pct={:.2} top5_new_pct={:.2} top5_delta_pp={:+.2}",
            dimension,
            segment_count,
            metric,
            fmt_num(base_total, 2),
            fmt_num(new_total, 2),
            signed_fmt_num(new_total - base_total, 2),
            top1_base,
            top1_new,
            top1_delta,
            top5_base,
            top5_new,
            top5_delta
        ));

        if let Some(top) = movers.first() {
            let segment = top
                .get("segment")
                .and_then(|x| x.as_str())
                .unwrap_or("(unknown)");
            let delta_metric = top
                .get("delta_primary_metric_value")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0);
            let delta_share = top
                .get("delta_share_pp")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0);
            out.push(format!(
                "top_level_strongest segment='{}' delta_{}={} delta_share_pp={:+.2}",
                segment,
                metric,
                fmt_num(delta_metric, 2),
                delta_share
            ));
        }
    }

    if let Some(changes) = v.get("major_global_changes").and_then(|x| x.as_array()) {
        for change in changes.iter().take(3) {
            let dim = change
                .get("dimension")
                .and_then(|x| x.as_str())
                .unwrap_or("(unknown)");
            let seg = change
                .get("segment")
                .and_then(|x| x.as_str())
                .unwrap_or("(unknown)");
            let metric = change
                .get("primary_metric")
                .and_then(|x| x.as_str())
                .unwrap_or("primary_metric");
            let delta_metric = change
                .get("delta_primary_metric_value")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0);
            let delta_share = change
                .get("delta_share_pp")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0);
            out.push(format!(
                "major_change dimension='{}' segment='{}' delta_{}={} delta_share_pp={:+.2}",
                dim,
                seg,
                metric,
                fmt_num(delta_metric, 2),
                delta_share
            ));
        }
    }

    for step in steps.iter().skip(1).take(4) {
        let depth = step.get("depth").and_then(|x| x.as_u64()).unwrap_or(0);
        let dimension = step
            .get("dimension")
            .and_then(|x| x.as_str())
            .unwrap_or("(unknown)");
        let metric = step
            .get("primary_metric")
            .and_then(|x| x.as_str())
            .unwrap_or("primary_metric");
        let scope =
            investigate_scope_to_text(step.get("scope").unwrap_or(&serde_json::Value::Null));
        let mover = step
            .get("movers")
            .and_then(|x| x.as_array())
            .and_then(|xs| xs.first());
        if let Some(top) = mover {
            let segment = top
                .get("segment")
                .and_then(|x| x.as_str())
                .unwrap_or("(unknown)");
            let delta_metric = top
                .get("delta_primary_metric_value")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0);
            let delta_share = top
                .get("delta_share_pp")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0);
            out.push(format!(
                "follow_up depth={} scope='{}' grouped_by='{}' strongest='{}' delta_{}={} delta_share_pp={:+.2}",
                depth,
                scope,
                dimension,
                segment,
                metric,
                fmt_num(delta_metric, 2),
                delta_share
            ));
        }
    }

    if let Some(stop) = v.get("stopping_reason").and_then(|x| x.as_str()) {
        out.push(format!("stop_reason='{}'", stop));
    }
    if let Some(next_q) = v.get("recommended_next_question").and_then(|x| x.as_str()) {
        out.push(format!("recommended_next='{}'", next_q));
    }
    out
}

fn strip_inline_markdown_noise(text: &str) -> String {
    // Keep evidence tags like [E1] but remove common emphasis/code markers.
    let mut out = text
        .replace("```", " ")
        .replace("**", "")
        .replace("__", "")
        .replace('`', "");
    if !out.contains("://") {
        out = out.replace("//", "");
    }
    out
}

fn sanitize_explain_analyze_answer(raw: &str, max_bullets: usize) -> String {
    let mut text = raw.trim();
    if let Some((_, tail)) = raw.rsplit_once("assistant") {
        let trimmed = tail.trim();
        if !trimmed.is_empty() {
            text = trimmed;
        }
    }

    let mut out = Vec::<String>::new();
    let mut seen = HashSet::<String>::new();
    for line in text.lines() {
        let cleaned = strip_inline_markdown_noise(line);
        let trimmed = cleaned
            .trim()
            .trim_start_matches('-')
            .trim_start_matches('*')
            .trim();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("question:")
            || lower.starts_with("analysis context:")
            || lower.starts_with("here is")
            || lower.starts_with("summary:")
            || lower.contains("```")
        {
            continue;
        }
        let alpha = trimmed.chars().filter(|c| c.is_ascii_alphabetic()).count();
        if alpha < 8 {
            continue;
        }
        let canonical = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
        let key = canonical.to_ascii_lowercase();
        if seen.insert(key) {
            out.push(format!("- {}", canonical));
        }
        if out.len() >= max_bullets.max(1) {
            break;
        }
    }
    out.join("\n")
}

fn explain_analyze_answer_is_usable(answer: &str, evidence: &[String]) -> bool {
    if answer.trim().is_empty() {
        return false;
    }
    if contains_long_decimal(answer, 2) {
        return false;
    }
    let has_citation = answer.contains("[E");
    if !has_citation {
        return false;
    }
    let mut allowed = Vec::<f64>::new();
    for line in evidence {
        allowed.extend(extract_numeric_values(line));
    }
    allowed.extend((1..=evidence.len()).map(|i| i as f64));
    if !summary_numbers_are_grounded(answer, &allowed) {
        return false;
    }
    true
}

fn deterministic_explain_analyze_from_evidence(
    _question: &str,
    evidence: &[String],
    max_bullets: usize,
) -> String {
    if evidence.is_empty() {
        return "- Unable to provide a grounded summary from the provided analysis context."
            .to_string();
    }
    evidence
        .iter()
        .take(max_bullets.max(1))
        .enumerate()
        .map(|(i, line)| format!("- {} [E{}]", line, i + 1))
        .collect::<Vec<_>>()
        .join("\n")
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
        "the", "and", "for", "of", "to", "in", "on", "with", "a", "an", "or", "by", "from", "per",
        "will", "assumes", "assume", "data",
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
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
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

fn signed_fmt_num(value: f64, decimals: usize) -> String {
    if value >= 0.0 {
        format!("+{}", fmt_num(value, decimals))
    } else {
        fmt_num(value, decimals)
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
