use anyhow::Result;
use factor_io::ArtifactSummary;

pub fn markdown_report(summary: &ArtifactSummary) -> Result<String> {
    let ev = &summary.model.explained_variance_ratio;
    let top_var = ev.iter().take(3).copied().collect::<Vec<_>>();
    let ev_cum_1 = ev.iter().take(1).sum::<f64>();
    let ev_cum_2 = ev.iter().take(2).sum::<f64>();
    let ev_cum_3 = ev.iter().take(3).sum::<f64>();
    let assets = summary.model.tickers.len();
    let k = summary.model.k;
    let near_zero_outliers = summary
        .top_outliers
        .iter()
        .filter(|o| o.abs_residual <= 1e-6)
        .count();

    let mut s = String::new();
    s.push_str("# FactorLens Report\n\n");

    s.push_str("## Executive Interpretation\n");
    s.push_str(&format!(
        "- Factor 1 explains {:.1}% of variance; top 2 explain {:.1}%; top 3 explain {:.1}%.\n",
        ev_cum_1 * 100.0,
        ev_cum_2 * 100.0,
        ev_cum_3 * 100.0
    ));
    if k >= assets {
        s.push_str(
            "- Warning: factors `k` is equal to or greater than number of assets, so PCA can nearly fully reconstruct returns.\n",
        );
        s.push_str("- Result: residual-based outliers will be near zero and not informative.\n");
    }
    if near_zero_outliers >= (summary.top_outliers.len() * 8 / 10) {
        s.push_str(
            "- Warning: most residual outliers are near zero; try fewer factors (e.g., `--k 1` or `--k 2`) for useful residual diagnostics.\n",
        );
    }
    s.push('\n');

    s.push_str("## Model Summary\n");
    s.push_str(&format!("- Observations: {}\n", summary.model.dates.len()));
    s.push_str(&format!("- Assets: {}\n", assets));
    s.push_str(&format!("- Factors (k): {}\n", summary.model.k));
    s.push_str(&format!(
        "- Top explained variance ratios: {:?}\n\n",
        top_var
    ));

    s.push_str("## Top Outlier Days (Residual)\n");
    for o in &summary.top_outliers {
        s.push_str(&format!(
            "- {} residual={:.6} abs={:.6}\n",
            o.date, o.residual, o.abs_residual
        ));
    }
    s.push('\n');

    s.push_str("## How To Make This Useful\n");
    s.push_str("- Re-run with fewer factors than assets (example: `--k 1` or `--k 2`).\n");
    s.push_str("- Use real portfolio weights/holdings instead of equal weights.\n");
    s.push_str("- For interpretable drivers, run known-factor regression mode with `factors regress` and a named `factors.csv` (for example: MKT/SMB/HML).\n");

    Ok(s)
}
