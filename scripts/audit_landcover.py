"""Fixed annual land-cover consistency audit for 50 patches, 2017-2025."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/final_analysis.json").read_text(encoding="utf-8"))
P = CFG["analysis"]; OUT = ROOT / "outputs"
PTS = pd.read_csv(ROOT / "data/processed/sampled_patch_sample_points.csv")
PATCHES = pd.read_csv(OUT / "sampled_patches.csv")
CLASSES = {1: "water", 2: "trees", 4: "flooded_vegetation", 5: "crops",
           7: "built", 8: "bare", 9: "snow_ice", 10: "clouds", 11: "rangeland"}
VEG = {2, 4, 11}; YEARS = P["years"]; AVAILABLE = set(P["annual_landcover_available_years"])

def query(point, year):
    item = f"51R-{year}"
    url = f"https://planetarycomputer.microsoft.com/api/data/v1/item/point/{point.lon},{point.lat}"
    params = [("collection", P["annual_landcover_collection"]), ("item", item), ("assets", "data")]
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=25,
                             headers={"User-Agent": "ShanghaiUrbanVegetation/1.0"})
            r.raise_for_status(); code = int(r.json()["values"][0])
            return {**point.to_dict(), "year": year, "item_id": item, "class_code": code,
                    "class_name": CLASSES.get(code, "unknown"), "request_status": "ok"}
        except Exception as exc:
            if attempt == 2:
                return {**point.to_dict(), "year": year, "item_id": item,
                        "class_code": np.nan, "class_name": "request_failed",
                        "request_status": f"{type(exc).__name__}: {exc}"}
            time.sleep(2 + attempt * 2)

checkpoint = ROOT / "data/processed/landcover_point_audit.partial.csv"
rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
done = {(str(r["sample_point_id"]), int(r["year"])) for r in rows}
jobs = [(p, y) for _, p in PTS.iterrows() for y in sorted(AVAILABLE)
        if (str(p.sample_point_id), y) not in done]
print({"annual_point_jobs": len(jobs), "resumed": len(rows)}, flush=True)
with ThreadPoolExecutor(max_workers=36) as executor:
    futures = [executor.submit(query, point, year) for point, year in jobs]
    for n, future in enumerate(as_completed(futures), 1):
        rows.append(future.result())
        if n % 100 == 0 or n == len(futures):
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(f"land-cover requests {n}/{len(futures)}", flush=True)
samples = pd.DataFrame(rows).drop_duplicates(["sample_point_id", "year"])
samples.to_csv(ROOT / "data/processed/landcover_point_audit.csv", index=False, encoding="utf-8-sig")

audit = []
for _, patch in PATCHES.iterrows():
    annual = {}
    for year in sorted(AVAILABLE):
        g = samples[(samples.patch_id == patch.patch_id) & (samples.year == year)]
        valid = g.dropna(subset=["class_code"]); expected = 2 if patch.vegetation_type == "tree" else 11
        exact = (valid.class_code == expected).mean() if len(valid) else np.nan
        veg = valid.class_code.isin(VEG).mean() if len(valid) else np.nan
        crop = (valid.class_code == 5).mean() if len(valid) else np.nan
        built = (valid.class_code == 7).mean() if len(valid) else np.nan
        majority = valid.class_name.mode().iloc[0] if len(valid) else "unavailable"
        persistent = len(valid) >= 2 and (exact >= 2 / 3 or veg >= 2 / 3) and crop <= 1 / 3 and built <= 1 / 3
        major_nonveg = len(valid) >= 2 and (built >= P["landcover_probable_conversion_built_fraction"] or (
            veg <= P["landcover_probable_conversion_max_vegetation_fraction"] and
            majority not in ["trees", "rangeland", "flooded_vegetation"]))
        annual[year] = dict(valid=valid, exact=exact, veg=veg, crop=crop, built=built,
                            majority=majority, persistent=persistent, major_nonveg=major_nonveg)
    conversion_year = None
    for year in sorted(AVAILABLE):
        prior_persistent = any(annual[y]["persistent"] for y in AVAILABLE if y < year)
        if prior_persistent and annual[year]["major_nonveg"] and annual.get(year + 1, {}).get("major_nonveg", False):
            conversion_year = year; break
    for year in YEARS:
        if year in AVAILABLE:
            a = annual[year]; valid = a["valid"]; exact = a["exact"]; veg = a["veg"]
            crop = a["crop"]; built = a["built"]; majority = a["majority"]
            if conversion_year is not None and year >= conversion_year: status = "probable_landcover_change"
            elif a["persistent"]: status = "persistent_vegetation"
            else: status = "uncertain_landcover"
            source = "Impact Observatory/Esri/Microsoft Annual LULC v2"
        else:
            valid = pd.DataFrame(); exact = veg = crop = built = np.nan; majority = "not_available"
            status = "uncertain_no_annual_map"; source = "no public annual v2 map after 2023"
        conversion_seen = conversion_year is not None and year >= conversion_year
        exclude_conversion = conversion_seen
        audit.append({"patch_id": patch.patch_id, "block_id": patch.block_id, "year": year,
                      "vegetation_type": patch.vegetation_type,
                      "sampled_vegetation_confidence": patch.vegetation_confidence,
                      "annual_landcover_source": source, "annual_item": f"51R-{year}" if year in AVAILABLE else "NA",
                      "valid_points": len(valid), "majority_class": majority,
                      "exact_type_agreement": exact, "vegetation_fraction": veg,
                      "crop_fraction": crop, "built_fraction": built,
                      "landcover_temporal_status": status,
                      "prior_or_current_probable_conversion": conversion_seen,
                      "exclude_probable_conversion": exclude_conversion,
                      "landcover_audit_inclusion": not exclude_conversion,
                      "landcover_exclusion_reason": "probable conversion at or before this year" if exclude_conversion else ""})
audit = pd.DataFrame(audit)
audit.to_csv(OUT / "patch_year_landcover_audit.csv", index=False, encoding="utf-8-sig")
summary = audit.groupby(["year", "landcover_temporal_status"]).size().rename("patch_years").reset_index()
summary.to_csv(OUT / "landcover_audit_summary.csv", index=False, encoding="utf-8-sig")
print(summary.to_string(index=False))
print({"rows": len(audit), "patches_with_conversion": audit[audit.exclude_probable_conversion].patch_id.nunique(),
       "failed_point_requests": int((samples.request_status != "ok").sum())})
