use anyhow::{anyhow, Result};
use chrono::NaiveDate;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

type PriceReturnMatrix = (Vec<NaiveDate>, Vec<String>, Vec<Vec<f64>>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub date: NaiveDate,
    pub ticker: String,
    pub close: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioWeight {
    pub ticker: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorModel {
    pub dates: Vec<NaiveDate>,
    pub tickers: Vec<String>,
    pub k: usize,
    pub explained_variance_ratio: Vec<f64>,
    pub factor_returns: Vec<Vec<f64>>, // rows: time, cols: factors
    pub loadings: Vec<Vec<f64>>,       // rows: assets, cols: factors
    pub standardized_returns: Vec<Vec<f64>>, // rows: time, cols: assets
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDay {
    pub date: NaiveDate,
    pub residual: f64,
    pub abs_residual: f64,
}

pub fn fit_pca(prices: &[PricePoint], k: usize) -> Result<FactorModel> {
    let (dates, tickers, returns) = prices_to_return_matrix(prices)?;
    let std_returns = standardize_columns(&returns)?;

    let rows = std_returns.len();
    let cols = std_returns[0].len();
    let flat: Vec<f64> = std_returns.iter().flat_map(|r| r.iter().copied()).collect();
    let x = DMatrix::from_row_slice(rows, cols, &flat);

    let cov = (&x.transpose() * &x) / (rows as f64 - 1.0);
    let eig = cov.symmetric_eigen();

    let mut eigen_pairs: Vec<(f64, Vec<f64>)> = eig
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let vec = eig
                .eigenvectors
                .column(i)
                .iter()
                .copied()
                .collect::<Vec<_>>();
            (val, vec)
        })
        .collect();

    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let k_eff = k.min(cols).max(1);
    let total_var: f64 = eigen_pairs.iter().map(|x| x.0.max(0.0)).sum();
    if total_var <= 0.0 {
        return Err(anyhow!("non-positive total variance"));
    }

    let top_vals: Vec<f64> = eigen_pairs
        .iter()
        .take(k_eff)
        .map(|x| x.0.max(0.0))
        .collect();
    let explained_variance_ratio = top_vals.iter().map(|v| v / total_var).collect::<Vec<_>>();

    let mut loading_matrix = DMatrix::<f64>::zeros(cols, k_eff);
    for (j, (_val, vec)) in eigen_pairs.iter().take(k_eff).enumerate() {
        for i in 0..cols {
            loading_matrix[(i, j)] = vec[i];
        }
    }

    let factor_ret = &x * &loading_matrix;

    let factor_returns = (0..rows)
        .map(|i| (0..k_eff).map(|j| factor_ret[(i, j)]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let loadings = (0..cols)
        .map(|i| {
            (0..k_eff)
                .map(|j| loading_matrix[(i, j)])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Ok(FactorModel {
        dates,
        tickers,
        k: k_eff,
        explained_variance_ratio,
        factor_returns,
        loadings,
        standardized_returns: std_returns,
    })
}

pub fn standardize_columns(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Err(anyhow!("empty matrix"));
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut out = vec![vec![0.0; cols]; rows];

    for j in 0..cols {
        let mean = matrix.iter().map(|r| r[j]).sum::<f64>() / rows as f64;
        let var = matrix
            .iter()
            .map(|r| {
                let d = r[j] - mean;
                d * d
            })
            .sum::<f64>()
            / (rows as f64 - 1.0).max(1.0);
        let std = var.sqrt();
        let denom = if std <= 1e-12 { 1.0 } else { std };

        for i in 0..rows {
            out[i][j] = (matrix[i][j] - mean) / denom;
        }
    }
    Ok(out)
}

pub fn portfolio_returns_from_weights(
    model: &FactorModel,
    weights: &[PortfolioWeight],
) -> Result<Vec<f64>> {
    let map = weights
        .iter()
        .map(|w| (w.ticker.clone(), w.weight))
        .collect::<HashMap<_, _>>();

    let mut wvec = vec![0.0; model.tickers.len()];
    for (i, t) in model.tickers.iter().enumerate() {
        if let Some(w) = map.get(t) {
            wvec[i] = *w;
        }
    }

    let sum_w: f64 = wvec.iter().sum();
    if sum_w.abs() <= 1e-12 {
        return Err(anyhow!("portfolio has no overlap with model tickers"));
    }

    for w in &mut wvec {
        *w /= sum_w;
    }

    let mut out = vec![0.0; model.standardized_returns.len()];
    for (i, row) in model.standardized_returns.iter().enumerate() {
        out[i] = row.iter().zip(wvec.iter()).map(|(r, w)| r * w).sum::<f64>();
    }
    Ok(out)
}

pub fn portfolio_factor_contributions(
    model: &FactorModel,
    weights: &[PortfolioWeight],
) -> Result<Vec<Vec<f64>>> {
    let map = weights
        .iter()
        .map(|w| (w.ticker.clone(), w.weight))
        .collect::<HashMap<_, _>>();

    let mut wvec = vec![0.0; model.tickers.len()];
    for (i, t) in model.tickers.iter().enumerate() {
        if let Some(w) = map.get(t) {
            wvec[i] = *w;
        }
    }

    let sum_w: f64 = wvec.iter().sum();
    if sum_w.abs() <= 1e-12 {
        return Err(anyhow!("portfolio has no overlap with model tickers"));
    }
    for w in &mut wvec {
        *w /= sum_w;
    }

    let k = model.k;
    let mut exposure = vec![0.0; k];
    for (asset_i, &w) in wvec.iter().enumerate() {
        for (f, e) in exposure.iter_mut().enumerate().take(k) {
            *e += w * model.loadings[asset_i][f];
        }
    }

    let mut contrib = vec![vec![0.0; k]; model.factor_returns.len()];
    for (t, fr) in model.factor_returns.iter().enumerate() {
        for f in 0..k {
            contrib[t][f] = exposure[f] * fr[f];
        }
    }
    Ok(contrib)
}

pub fn compute_outliers(
    dates: &[NaiveDate],
    portfolio_returns: &[f64],
    factor_contrib: &[Vec<f64>],
    top_n: usize,
) -> Vec<OutlierDay> {
    let mut out = Vec::with_capacity(dates.len());
    for i in 0..dates.len() {
        let fit = factor_contrib[i].iter().sum::<f64>();
        let residual = portfolio_returns[i] - fit;
        out.push(OutlierDay {
            date: dates[i],
            residual,
            abs_residual: residual.abs(),
        });
    }

    out.sort_by(|a, b| {
        b.abs_residual
            .partial_cmp(&a.abs_residual)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(top_n);
    out
}

fn prices_to_return_matrix(prices: &[PricePoint]) -> Result<PriceReturnMatrix> {
    if prices.is_empty() {
        return Err(anyhow!("no prices"));
    }

    let mut by_ticker: BTreeMap<String, Vec<(NaiveDate, f64)>> = BTreeMap::new();
    for p in prices {
        by_ticker
            .entry(p.ticker.clone())
            .or_default()
            .push((p.date, p.close));
    }

    let tickers = by_ticker.keys().cloned().collect::<Vec<_>>();
    if tickers.len() < 2 {
        return Err(anyhow!("need at least 2 tickers"));
    }

    let mut ret_maps = Vec::new();
    for (_ticker, mut points) in by_ticker {
        points.sort_by_key(|x| x.0);
        let mut map = HashMap::new();
        for w in points.windows(2) {
            let (d0, p0) = w[0];
            let (d1, p1) = w[1];
            if d1 <= d0 || p0 <= 0.0 {
                continue;
            }
            map.insert(d1, (p1 / p0).ln());
        }
        ret_maps.push(map);
    }

    let mut common_dates = ret_maps[0].keys().copied().collect::<Vec<_>>();
    common_dates.sort();
    common_dates.retain(|d| ret_maps.iter().all(|m| m.contains_key(d)));

    if common_dates.len() < 20 {
        return Err(anyhow!("not enough overlapping return rows (need >= 20)"));
    }

    let mut matrix = vec![vec![0.0; tickers.len()]; common_dates.len()];
    for (i, d) in common_dates.iter().enumerate() {
        for (j, m) in ret_maps.iter().enumerate() {
            matrix[i][j] = *m.get(d).ok_or_else(|| anyhow!("missing return"))?;
        }
    }

    Ok((common_dates, tickers, matrix))
}
