"""Original-sample endpoint, robustness, sensitivity, and thermal analyses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/final_analysis.json").read_text(encoding="utf-8"))
P = CFG["analysis"]; OUT = ROOT / "outputs"; RNG = np.random.default_rng(CFG["random_seed"])
PATCHES = pd.read_csv(OUT / "sampled_patches.csv")
BLOCKS = pd.read_csv(OUT / "sampled_blocks.csv")
LC = pd.read_csv(OUT / "patch_year_landcover_audit.csv")
DENSITY = pd.read_csv(OUT / "observation_density.csv")
CALYEAR = pd.read_csv(OUT / "landsat_s2_calibration_by_year.csv")
HOT = pd.read_csv(OUT / "hot_year_classification.csv")
THERMAL = pd.read_csv(OUT / "thermal_metrics.csv")
ENDPOINTS = ["peak_NDVI", "summer_NDVI", "seasonal_amplitude", "summer_decline", "autumn_NDVI"]
PRIMARY = ENDPOINTS[:2]; X = "built_fraction_1000m"; CROP = "crop_context_1000m"
STRATUM = BLOCKS.set_index("block_id").urbanization_stratum.to_dict()
def as_bool(v): return str(v).lower() == "true"
LC["exclude_probable_conversion"] = LC.exclude_probable_conversion.map(as_bool)
CALFLAG = CALYEAR.set_index("year").calibration_instability_flag.map(as_bool).to_dict()

def endpoint_table(path, series):
    ts = pd.read_csv(path); ts.date = pd.to_datetime(ts.date); rows = []
    for _, patch in PATCHES.iterrows():
        for year in P["years"]:
            g = ts[(ts.patch_id == patch.patch_id) & (ts.year == year)].sort_values("date")
            d = DENSITY[(DENSITY.series == series) & (DENSITY.patch_id == patch.patch_id) & (DENSITY.year == year)].iloc[0]
            lc = LC[(LC.patch_id == patch.patch_id) & (LC.year == year)].iloc[0]
            active = g[g.date.dt.month.isin(P["active_months"])]
            summer = g[g.date.dt.month.isin(P["summer_months"])]
            autumn = g[g.date.dt.month.isin(P["autumn_months"])]
            peak = active.ndvi.quantile(.9) if len(active) else np.nan
            low = active.ndvi.quantile(.1) if len(active) else np.nan
            summer_ndvi = summer.ndvi.median() if len(summer) else np.nan
            autumn_ndvi = autumn.ndvi.median() if len(autumn) else np.nan
            reasons = []
            if int(d.n_valid_observations) < P["minimum_patch_year_observations"]: reasons.append("insufficient_observations")
            if int(d.summer_observations) < P["minimum_summer_observations"]: reasons.append("insufficient_summer_observations")
            if pd.isna(d.max_active_season_gap_days) or d.max_active_season_gap_days > P["maximum_active_season_gap_days"]: reasons.append("active_season_gap")
            temporal = len(reasons) == 0
            if patch.vegetation_confidence not in ["HIGH", "MEDIUM"]: reasons.append("low_sampled_landcover_confidence")
            if lc.exclude_probable_conversion: reasons.append("probable_landcover_conversion")
            rows.append({"series": series, "patch_id": patch.patch_id, "block_id": patch.block_id,
                "year": year, "vegetation_type": patch.vegetation_type,
                "urbanization_stratum": STRATUM.get(patch.block_id),
                "vegetation_confidence": patch.vegetation_confidence,
                "landcover_temporal_status": lc.landcover_temporal_status,
                "calibration_instability_flag": CALFLAG.get(year, False),
                X: patch.built_fraction_1000m, "built_fraction_90m": patch.built_fraction_90m,
                CROP: patch.crop_context_1000m, "patch_area_ha": patch.patch_area_ha,
                "n_observations": int(d.n_valid_observations), "summer_observations": int(d.summer_observations),
                "max_active_season_gap_days": d.max_active_season_gap_days,
                "temporal_qc_pass": temporal, "landcover_audit_inclusion": not lc.exclude_probable_conversion,
                "included_primary": len(reasons) == 0, "exclusion_reason": ";".join(reasons),
                "peak_NDVI": peak, "summer_NDVI": summer_ndvi,
                "seasonal_amplitude": peak - low if np.isfinite(peak) and np.isfinite(low) else np.nan,
                "summer_decline": peak - summer_ndvi if np.isfinite(peak) and np.isfinite(summer_ndvi) else np.nan,
                "autumn_NDVI": autumn_ndvi})
    return pd.DataFrame(rows)

combined = endpoint_table(ROOT / "data/processed/combined_patch_timeseries.csv", "S2+Landsat-fill")
s2only = endpoint_table(ROOT / "data/processed/s2_patch_timeseries_harmonized.csv", "S2-only")
combined.to_csv(OUT / "seasonal_metrics.csv", index=False, encoding="utf-8-sig")
qc_cols = ["series", "patch_id", "block_id", "year", "vegetation_type", "urbanization_stratum",
    "vegetation_confidence", "landcover_temporal_status", "calibration_instability_flag", CROP,
    "n_observations", "summer_observations", "max_active_season_gap_days", "temporal_qc_pass",
    "landcover_audit_inclusion", "included_primary", "exclusion_reason"]
both = pd.concat([combined, s2only], ignore_index=True)
both[qc_cols].to_csv(OUT / "qc.csv", index=False, encoding="utf-8-sig")
for name, keys in [("year", ["series", "year"]), ("vegetation_type", ["series", "vegetation_type"]),
                   ("urbanization_stratum", ["series", "urbanization_stratum"]), ("block", ["series", "block_id"])]:
    q = both.groupby(keys).agg(patch_years=("patch_id", "size"), temporal_qc_retained=("temporal_qc_pass", "sum"),
        primary_retained=("included_primary", "sum"), patches=("patch_id", "nunique")).reset_index()
    q.to_csv(OUT / f"qc_retention_by_{name}.csv", index=False, encoding="utf-8-sig")

main = combined[combined.included_primary].copy(); s2main = s2only[s2only.included_primary].copy()
def desc(df, endpoint, x=X):
    q = df.dropna(subset=[endpoint, x]); rho = pval = np.nan
    if len(q) >= 4 and q[x].nunique() > 1:
        s = spearmanr(q[x], q[endpoint]); rho, pval = s.statistic, s.pvalue
    return {"spearman_rho": rho, "spearman_p": pval}
def fit(df, endpoint, x=X, year_fe=True):
    q = df.dropna(subset=[endpoint, x, CROP, "vegetation_type", "block_id"]).copy()
    formula = f"{endpoint} ~ {x} + C(vegetation_type)" + (" + C(year)" if year_fe else "") + f" + {CROP}"
    if len(q) < 10 or q.block_id.nunique() < 4 or q[x].nunique() < 2:
        return None, {"formula": formula, "estimate": np.nan, "standard_error": np.nan, "ci_low": np.nan,
            "ci_high": np.nan, "p_value": np.nan, "effect_direction": "not_estimable", "n_patch_years": len(q),
            "n_patches": q.patch_id.nunique(), "n_blocks": q.block_id.nunique()}
    f = smf.ols(formula, q).fit(cov_type="cluster", cov_kwds={"groups": q.block_id})
    est, se = f.params.get(x, np.nan), f.bse.get(x, np.nan)
    return f, {"formula": formula, "estimate": est, "standard_error": se, "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se, "p_value": f.pvalues.get(x, np.nan),
        "effect_direction": "negative" if est < 0 else "positive", "n_patch_years": len(q),
        "n_patches": q.patch_id.nunique(), "n_blocks": q.block_id.nunique()}
def block_boot(df, endpoint, x=X):
    q = df.dropna(subset=[endpoint, x, CROP]); blocks = q.block_id.unique(); vals = []
    for _ in range(P["block_bootstrap_replicates"]):
        parts = []
        for i, block in enumerate(RNG.choice(blocks, len(blocks), replace=True)):
            a = q[q.block_id == block].copy(); a["_boot"] = f"{block}_{i}"; parts.append(a)
        a = pd.concat(parts, ignore_index=True)
        try: vals.append(smf.ols(f"{endpoint} ~ {x} + C(vegetation_type) + C(year) + {CROP}", a).fit().params.get(x, np.nan))
        except Exception: pass
    vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
    return (np.quantile(vals, .025), np.quantile(vals, .975), len(vals)) if len(vals) else (np.nan, np.nan, 0)

rows = []
for endpoint in PRIMARY:
    _, r = fit(main, endpoint); lo, hi, n = block_boot(main, endpoint)
    rows.append({"endpoint": endpoint, "predictor": X, **desc(main, endpoint), **r,
        "block_bootstrap_ci_low": lo, "block_bootstrap_ci_high": hi, "block_bootstrap_replicates": n})
primary_models = pd.DataFrame(rows)
primary_models.to_csv(OUT / "primary_models.csv", index=False, encoding="utf-8-sig")
secondary = []
for endpoint in ENDPOINTS[2:]:
    _, r = fit(main, endpoint); secondary.append({"endpoint": endpoint, "status": "secondary", **desc(main, endpoint), **r})
pd.DataFrame(secondary).to_csv(OUT / "secondary_models.csv", index=False, encoding="utf-8-sig")

yrows = []
for endpoint in PRIMARY:
    for year in P["years"]:
        q = main[main.year == year]; _, r = fit(q, endpoint, year_fe=False)
        yrows.append({"endpoint": endpoint, "year": year, **desc(q, endpoint), **r})
year_results = pd.DataFrame(yrows)
year_results.to_csv(OUT / "year_specific_results.csv", index=False, encoding="utf-8-sig")

hetero = []
for endpoint in PRIMARY:
    q = main.dropna(subset=[endpoint, X, CROP])
    f = smf.ols(f"{endpoint} ~ {X} * C(year) + C(vegetation_type) + {CROP}", q).fit(
        cov_type="cluster", cov_kwds={"groups": q.block_id})
    terms = [t for t in f.params.index if t.startswith(f"{X}:C(year)")]
    R = np.zeros((len(terms), len(f.params)))
    for i, term in enumerate(terms): R[i, list(f.params.index).index(term)] = 1
    joint_p = float(np.asarray(f.wald_test(R, scalar=True).pvalue).ravel()[0]) if terms else np.nan
    yr = year_results[year_results.endpoint == endpoint].dropna(subset=["estimate"])
    effect_range = yr.estimate.max() - yr.estimate.min(); trend = linregress(yr.year, yr.estimate) if len(yr) >= 3 else None
    material = bool(joint_p < .05 and effect_range > P["temporal_heterogeneity_effect_range_threshold"])
    pattern = "stable"
    if material:
        pattern = ("strengthening" if trend.slope < 0 else "weakening") if trend and trend.pvalue < .05 else "highly_variable"
    hetero.append({"endpoint": endpoint, "joint_interaction_p": joint_p, "annual_estimate_min": yr.estimate.min(),
        "annual_estimate_max": yr.estimate.max(), "annual_estimate_range": effect_range,
        "linear_slope_change_per_year": trend.slope if trend else np.nan, "linear_trend_p": trend.pvalue if trend else np.nan,
        "material_temporal_variation": material, "temporal_pattern": pattern})
pd.DataFrame(hetero).to_csv(OUT / "temporal_heterogeneity.csv", index=False, encoding="utf-8-sig")

tgrows = []
for endpoint in PRIMARY:
    q = main.dropna(subset=[endpoint, X, CROP]); formula = f"{endpoint} ~ {X} * C(vegetation_type) + C(year) + {CROP}"
    f = smf.ols(formula, q).fit(cov_type="cluster", cov_kwds={"groups": q.block_id})
    term = f"{X}:C(vegetation_type)[T.tree]"; grass = f.params[X]; delta = f.params.get(term, np.nan)
    cov = f.cov_params(); tree = grass + delta
    tree_se = np.sqrt(cov.loc[X, X] + cov.loc[term, term] + 2 * cov.loc[X, term])
    tgrows.append({"endpoint": endpoint, "formula": formula, "grass_slope": grass, "grass_se": f.bse[X],
        "grass_ci_low": grass - 1.96 * f.bse[X], "grass_ci_high": grass + 1.96 * f.bse[X],
        "tree_slope": tree, "tree_se": tree_se, "tree_ci_low": tree - 1.96 * tree_se,
        "tree_ci_high": tree + 1.96 * tree_se, "interaction_tree_minus_grass": delta,
        "interaction_se": f.bse.get(term, np.nan), "interaction_p": f.pvalues.get(term, np.nan),
        "n_patch_years": len(q), "n_patches": q.patch_id.nunique(), "n_blocks": q.block_id.nunique()})
pd.DataFrame(tgrows).to_csv(OUT / "tree_grass_results.csv", index=False, encoding="utf-8-sig")

lorows = []
for endpoint in PRIMARY:
    full = primary_models.loc[primary_models.endpoint == endpoint, "estimate"].iloc[0]
    for block in sorted(main.block_id.unique()):
        _, r = fit(main[main.block_id != block], endpoint)
        lorows.append({"endpoint": endpoint, "omitted_block": block, **r,
            "change_from_full": r["estimate"] - full,
            "relative_absolute_change": abs(r["estimate"] - full) / abs(full) if full else np.nan,
            "sign_reversal": np.sign(r["estimate"]) != np.sign(full)})
pd.DataFrame(lorows).to_csv(OUT / "leave_one_block_out.csv", index=False, encoding="utf-8-sig")

srows = []
for endpoint in PRIMARY:
    analyses = [("HIGH_MEDIUM_primary", main, X), ("HIGH_only", main[main.vegetation_confidence == "HIGH"], X),
        ("exclude_crop_context_ge_0.5", main[main[CROP] < .5], X), ("local_90m", main, "built_fraction_90m"),
        ("S2_only_HIGH_MEDIUM", s2main, X)]
    for label, data, x in analyses:
        _, r = fit(data, endpoint, x=x)
        srows.append({"analysis": label, "endpoint": endpoint, "predictor": x, **desc(data, endpoint, x), **r})
pd.DataFrame(srows).to_csv(OUT / "landcover_sensitivity.csv", index=False, encoding="utf-8-sig")

contrasts = []; levels = P["effect_contrast_built_levels"]
labels = {levels[0]: "low", levels[1]: "medium", levels[2]: "high"}
for _, r in primary_models.iterrows():
    for level in levels:
        dx = level - levels[0]; value = r.estimate * dx; se = r.standard_error * abs(dx)
        contrasts.append({"endpoint": r.endpoint, "reference_built_fraction": levels[0],
            "comparison_built_fraction": level, "built_context": labels[level],
            "predicted_difference_from_low": value, "standard_error": se,
            "ci_low": value - 1.96 * se, "ci_high": value + 1.96 * se})
pd.DataFrame(contrasts).to_csv(OUT / "effect_size_contrasts.csv", index=False, encoding="utf-8-sig")

hot = HOT.copy(); hot["hot_year"] = hot.hot_year.map(as_bool).astype(int)
thermal_patch = main.merge(THERMAL[["block_id", "year", "summer_median_lst_c", "n_valid_summer_scenes"]],
    on=["block_id", "year"], how="left").merge(hot[["year", "thermal_year_class", "hot_year"]], on="year", how="left")
trows = []
def thermal_fit(name, formula, data, term):
    q = data.dropna(subset=[term.split(":")[0]]); f = smf.ols(formula, q).fit(cov_type="cluster", cov_kwds={"groups": q.block_id})
    est, se = f.params.get(term, np.nan), f.bse.get(term, np.nan)
    trows.append({"model": name, "formula": formula, "term": term, "estimate": est, "standard_error": se,
        "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se, "p_value": f.pvalues.get(term, np.nan),
        "n": len(q), "n_blocks": q.block_id.nunique()})
thermal_fit("built_to_summer_lst", "summer_median_lst_c ~ block_built_fraction + C(year)",
    THERMAL.dropna(subset=["summer_median_lst_c"]), "block_built_fraction")
thermal_fit("lst_to_summer_ndvi", f"summer_NDVI ~ summer_median_lst_c + {X} + C(vegetation_type) + C(year) + {CROP}",
    thermal_patch, "summer_median_lst_c")
thermal_fit("lst_x_vegetation_type", f"summer_NDVI ~ summer_median_lst_c * C(vegetation_type) + {X} + C(year) + {CROP}",
    thermal_patch, "summer_median_lst_c:C(vegetation_type)[T.tree]")
thermal_fit("hot_year_x_built", f"summer_NDVI ~ {X} + {X}:hot_year + C(vegetation_type) + C(year) + {CROP}",
    thermal_patch, f"{X}:hot_year")
pd.DataFrame(trows).to_csv(OUT / "thermal_models.csv", index=False, encoding="utf-8-sig")
thermal_patch.to_csv(OUT / "patch_year_thermal_join.csv", index=False, encoding="utf-8-sig")

print(primary_models.to_string(index=False))
print(pd.DataFrame(hetero).to_string(index=False))
print(pd.DataFrame(tgrows).to_string(index=False))
print(pd.DataFrame(trows).to_string(index=False))
print({"primary_patch_years": len(main), "patches": main.patch_id.nunique(), "blocks": main.block_id.nunique(),
       "s2_only_primary_patch_years": len(s2main)})
