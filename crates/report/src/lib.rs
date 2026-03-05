use anyhow::Result;
use factor_io::ArtifactSummary;

pub fn markdown_report(summary: &ArtifactSummary) -> Result<String> {
    let ev = &summary.model.explained_variance_ratio;
    let top_var = ev.iter().take(3).copied().collect::<Vec<_>>();

    let mut s = String::new();
    s.push_str("# FactorLens Report\n\n");
    s.push_str("## Model Summary\n");
    s.push_str(&format!("- Observations: {}\n", summary.model.dates.len()));
    s.push_str(&format!("- Assets: {}\n", summary.model.tickers.len()));
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

    Ok(s)
}
