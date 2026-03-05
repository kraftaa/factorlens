use anyhow::{anyhow, Result};
use chrono::NaiveDate;
use csv::{ReaderBuilder, WriterBuilder};
use factor_core::{FactorModel, OutlierDay, PortfolioWeight, PricePoint};
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct PriceRow {
    date: String,
    ticker: String,
    close: f64,
}

#[derive(Debug, Deserialize)]
struct WeightRow {
    ticker: String,
    weight: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactSummary {
    pub model: FactorModel,
    pub top_outliers: Vec<OutlierDay>,
}

pub fn read_prices_csv(path: &Path) -> Result<Vec<PricePoint>> {
    let mut rdr = ReaderBuilder::new().trim(csv::Trim::All).from_path(path)?;
    let mut out = Vec::new();

    for row in rdr.deserialize::<PriceRow>() {
        let row = row?;
        let date = NaiveDate::parse_from_str(&row.date, "%Y-%m-%d")?;
        out.push(PricePoint {
            date,
            ticker: row.ticker,
            close: row.close,
        });
    }

    if out.is_empty() {
        return Err(anyhow!("prices file is empty"));
    }
    Ok(out)
}

pub fn read_portfolio_csv(path: &Path) -> Result<Vec<PortfolioWeight>> {
    let mut rdr = ReaderBuilder::new().trim(csv::Trim::All).from_path(path)?;
    let mut out = Vec::new();

    for row in rdr.deserialize::<WeightRow>() {
        let row = row?;
        out.push(PortfolioWeight {
            ticker: row.ticker,
            weight: row.weight,
        });
    }

    if out.is_empty() {
        return Err(anyhow!("portfolio file is empty"));
    }
    Ok(out)
}

pub fn write_factor_artifacts(
    out_dir: &Path,
    model: &FactorModel,
    contributions: &[Vec<f64>],
    outliers: &[OutlierDay],
) -> Result<()> {
    create_dir_all(out_dir)?;

    let summary = ArtifactSummary {
        model: model.clone(),
        top_outliers: outliers.to_vec(),
    };
    let json_path = out_dir.join("factors.json");
    serde_json::to_writer_pretty(File::create(json_path)?, &summary)?;

    write_exposures_csv(out_dir.join("exposures.csv"), model)?;
    write_attribution_csv(out_dir.join("attribution.csv"), &model.dates, contributions)?;
    write_outliers_csv(out_dir.join("outliers.csv"), outliers)?;
    Ok(())
}

pub fn read_artifact_summary(out_dir: &Path) -> Result<ArtifactSummary> {
    let path = out_dir.join("factors.json");
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let summary = serde_json::from_reader(reader)?;
    Ok(summary)
}

pub fn list_artifact_paths(out_dir: &Path) -> Vec<PathBuf> {
    vec![
        out_dir.join("factors.json"),
        out_dir.join("exposures.csv"),
        out_dir.join("attribution.csv"),
        out_dir.join("outliers.csv"),
    ]
}

fn write_exposures_csv(path: PathBuf, model: &FactorModel) -> Result<()> {
    let mut wtr = WriterBuilder::new().from_path(path)?;
    let mut header = vec!["ticker".to_string()];
    header.extend((1..=model.k).map(|i| format!("factor_{}", i)));
    wtr.write_record(header)?;

    for (ticker, loadings) in model.tickers.iter().zip(model.loadings.iter()) {
        let mut rec = vec![ticker.to_string()];
        rec.extend(loadings.iter().map(|x| format!("{:.8}", x)));
        wtr.write_record(rec)?;
    }

    wtr.flush()?;
    Ok(())
}

fn write_attribution_csv(
    path: PathBuf,
    dates: &[NaiveDate],
    contributions: &[Vec<f64>],
) -> Result<()> {
    if contributions.is_empty() {
        return Err(anyhow!("empty contributions"));
    }

    let k = contributions[0].len();
    let mut wtr = WriterBuilder::new().from_path(path)?;
    let mut header = vec!["date".to_string()];
    header.extend((1..=k).map(|i| format!("factor_{}_contrib", i)));
    header.push("total_factor_contrib".to_string());
    wtr.write_record(header)?;

    for (d, row) in dates.iter().zip(contributions.iter()) {
        let mut rec = vec![d.to_string()];
        rec.extend(row.iter().map(|x| format!("{:.8}", x)));
        rec.push(format!("{:.8}", row.iter().sum::<f64>()));
        wtr.write_record(rec)?;
    }

    wtr.flush()?;
    Ok(())
}

fn write_outliers_csv(path: PathBuf, outliers: &[OutlierDay]) -> Result<()> {
    let mut wtr = WriterBuilder::new().from_path(path)?;
    wtr.write_record(["date", "residual", "abs_residual"])?;

    for o in outliers {
        wtr.write_record([
            o.date.to_string(),
            format!("{:.8}", o.residual),
            format!("{:.8}", o.abs_residual),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}
