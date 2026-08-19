""" platform/window sensor audit. Does not read vegetation endpoints."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
OUT = ROOT / "outputs"
DATA = ROOT / "data" / "processed"
CFG = json.loads((ROOT / "config" / "final_analysis.json").read_text(encoding="utf-8"))
Q = CFG["replication"]["calibration_quality"]

s2 = pd.read_csv(DATA / "s2_patch_timeseries_harmonized.csv", parse_dates=["date"])
ls = pd.read_csv(DATA / "landsat_patch_timeseries_raw.csv", parse_dates=["date"])
s2["s2_platform"] = s2.item_id.str.extract(r"^(S2[AB])")
ls["landsat_platform"] = ls.item_id.str.extract(r"^(L[CET]0[789])")
ls["landsat_platform"] = ls.landsat_platform.replace({"LC08":"Landsat 8", "LC09":"Landsat 9", "LE07":"Landsat 7"})

rows = []
for lr in ls.itertuples(index=False):
    q = s2[(s2.patch_id == lr.patch_id) & (s2.year == lr.year)].copy()
    if q.empty:
        continue
    q["signed_delta_days"] = (q.date - lr.date).dt.days
    q["delta_days"] = q.signed_delta_days.abs()
    q = q[q.delta_days <= 3].sort_values(["delta_days", "date"])
    if q.empty:
        continue
    sr = q.iloc[0]
    rows.append({
        "patch_id": lr.patch_id, "block_id": lr.block_id, "date_landsat": lr.date.date(),
        "date_sentinel2": sr.date.date(), "year": int(lr.year), "vegetation_type": lr.vegetation_type,
        "s2_platform": sr.s2_platform, "landsat_platform": lr.landsat_platform,
        "sentinel2_item_id": sr.item_id, "landsat_item_id": lr.item_id,
        "time_difference_days": int(sr.signed_delta_days), "absolute_time_difference_days": int(sr.delta_days),
        "s2_ndvi": float(sr.ndvi), "landsat_ndvi": float(lr.ndvi_raw),
        "s2_valid_sample_points": int(sr.valid_sample_points), "landsat_valid_sample_points": int(lr.valid_sample_points),
        "atmospheric_qa_metadata": "S2 SCL clear categories and Landsat QA_PIXEL bits 0-5 applied upstream; scene-level values retained in raw point tables",
    })

pairs = pd.DataFrame(rows).drop_duplicates(["patch_id", "landsat_item_id", "sentinel2_item_id"])
pairs.to_csv(TABLES/"sensor_matched_pairs.csv", index=False, encoding="utf-8-sig")

def fit(d):
    x, y = d.landsat_ndvi.to_numpy(), d.s2_ndvi.to_numpy()
    X = np.column_stack([np.ones(len(d)), x, (d.vegetation_type == "tree").astype(float)])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    residual = y - pred
    by_type = pd.DataFrame({"type": d.vegetation_type.to_numpy(), "resid": residual}).groupby("type").resid.mean()
    return {
        "intercept": coef[0], "slope": coef[1], "tree_offset": coef[2],
        "rmse": np.sqrt(np.mean(residual**2)), "bias": np.mean(residual),
        "grass_residual_bias": by_type.get("grass", np.nan), "tree_residual_bias": by_type.get("tree", np.nan),
    }

platform_rows = []
for (sp, lp), d in pairs.groupby(["s2_platform", "landsat_platform"], dropna=False):
    f = fit(d)
    year_stats = []
    for year, yd in d.groupby("year"):
        if len(yd) >= 10:
            yf = fit(yd); year_stats.append((year, len(yd), yf["slope"], yf["rmse"]))
    annual_slope_dev = max([abs(z[2] - f["slope"]) for z in year_stats], default=np.nan)
    annual_rmse_ratio = max([z[3] / f["rmse"] for z in year_stats], default=np.nan)
    reasons = []
    if len(d) < Q["minimum_platform_pairs"]: reasons.append("insufficient_pairs")
    if f["rmse"] > Q["maximum_rmse"]: reasons.append("rmse")
    if abs(f["bias"]) > Q["maximum_absolute_bias"]: reasons.append("bias")
    if not Q["minimum_slope"] <= f["slope"] <= Q["maximum_slope"]: reasons.append("slope")
    if max(abs(f["grass_residual_bias"]), abs(f["tree_residual_bias"])) > Q["maximum_type_residual_bias"]: reasons.append("type_bias")
    if d.year.nunique() < Q["minimum_years"]: reasons.append("year_coverage")
    if np.isfinite(annual_slope_dev) and annual_slope_dev > Q["annual_slope_difference_flag"]: reasons.append("annual_slope_stability")
    if np.isfinite(annual_rmse_ratio) and annual_rmse_ratio > Q["annual_rmse_ratio_flag"]: reasons.append("annual_rmse_stability")
    platform_rows.append({"s2_platform": sp, "landsat_platform": lp, "n_pairs": len(d),
        "n_years": d.year.nunique(), **f, "maximum_annual_slope_deviation": annual_slope_dev,
        "maximum_annual_rmse_ratio": annual_rmse_ratio, "quality_pass": len(reasons) == 0,
        "quality_flags": ";".join(reasons), "years": ";".join(map(str, sorted(d.year.unique())))})
platform = pd.DataFrame(platform_rows)
platform.to_csv(TABLES/"platform_calibration.csv", index=False, encoding="utf-8-sig")

window_rows = []
for window in [0, 1, 3]:
    for (sp, lp), d0 in pairs[pairs.absolute_time_difference_days <= window].groupby(["s2_platform", "landsat_platform"]):
        f = fit(d0)
        window_rows.append({"window_days": window, "s2_platform": sp, "landsat_platform": lp,
                            "n_pairs": len(d0), "n_years": d0.year.nunique(), **f})
windows = pd.DataFrame(window_rows)
if len(windows):
    ranges = windows.groupby(["s2_platform", "landsat_platform"]).slope.agg(lambda x: x.max()-x.min()).rename("window_slope_range")
    windows = windows.merge(ranges, on=["s2_platform", "landsat_platform"], how="left")
    windows["window_stability_flag"] = windows.window_slope_range > Q["window_slope_range_flag"]
windows.to_csv(TABLES/"calibration_window_sensitivity.csv", index=False, encoding="utf-8-sig")

print({"matched_pairs": len(pairs), "platform_groups": len(platform), "groups_passing_all_thresholds": int(platform.quality_pass.sum())})
print(platform.to_string(index=False))
