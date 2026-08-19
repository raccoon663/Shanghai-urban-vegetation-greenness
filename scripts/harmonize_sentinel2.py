"""Correct Sentinel-2 L2A PB04+ radiometric offset without overwriting raw tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/final_analysis.json").read_text(encoding="utf-8"))
P = CFG["analysis"]; audit = []; all_patch = []
cutoff = pd.Timestamp(P["sentinel2_baseline_offset_start"])
offset = P["sentinel2_new_baseline_reflectance_offset"]
for year in P["years"]:
    raw = pd.read_csv(ROOT / f"data/processed/s2_point_samples_{year}.csv")
    h = raw.copy(); h["date"] = pd.to_datetime(h.date)
    h["red_raw_scaled"] = h.red; h["nir_raw_scaled"] = h.nir
    h["pb04_offset_removed"] = h.date >= cutoff
    h.loc[h.pb04_offset_removed, "red"] = h.loc[h.pb04_offset_removed, "red"] - offset
    h.loc[h.pb04_offset_removed, "nir"] = h.loc[h.pb04_offset_removed, "nir"] - offset
    h.to_csv(ROOT / f"data/processed/s2_point_samples_harmonized_{year}.csv", index=False)
    patch_rows = []
    for (patch_id, date, item_id), group in h.groupby(["patch_id", "date", "item_id"]):
        good = group[group.clear.astype(str).str.lower().eq("true") & (group.red > 0) & (group.nir > 0)]
        if len(good) >= P["minimum_clear_sample_points"]:
            red, nir = good.red.median(), good.nir.median(); base = group.iloc[0]
            patch_rows.append({"patch_id": patch_id, "block_id": base.block_id,
                "vegetation_type": base.vegetation_type, "date": pd.Timestamp(date).date(),
                "year": year, "item_id": item_id, "valid_sample_points": len(good),
                "ndvi": (nir - red) / (nir + red), "source": "Sentinel-2 L2A PB04-harmonized"})
    patch = pd.DataFrame(patch_rows)
    patch.to_csv(ROOT / f"data/processed/s2_patch_timeseries_harmonized_{year}.csv", index=False)
    all_patch.extend(patch_rows)
    old = pd.read_csv(ROOT / f"data/processed/s2_patch_timeseries_{year}.csv")
    audit.append({"year": year, "raw_point_rows": len(raw),
        "point_rows_offset_corrected": int(h.pb04_offset_removed.sum()),
        "raw_patch_dates": len(old), "harmonized_patch_dates": len(patch),
        "raw_median_ndvi": old.ndvi.median() if len(old) else np.nan,
        "harmonized_median_ndvi": patch.ndvi.median() if len(patch) else np.nan})
pd.DataFrame(all_patch).to_csv(ROOT / "data/processed/s2_patch_timeseries_harmonized.csv", index=False)
pd.DataFrame(audit).to_csv(ROOT / "outputs/s2_baseline_harmonization_audit.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(audit).to_string(index=False))
