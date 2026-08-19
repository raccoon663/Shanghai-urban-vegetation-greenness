# Methods

This document describes the analysis pipeline that produced the tables in `outputs/tables/` and the figures in `assets/figures/`. It is a description of the study protocol, not a development log.

## Scientific question and endpoints

**Question:** How is surrounding urbanization intensity associated with summer vegetation greenness across Shanghai tree and grass patches?

- **Primary confirmatory endpoint:** summer NDVI — median valid NDVI during July–August.
- **Secondary endpoint:** peak NDVI — 90th percentile of valid March–November NDVI.
- Excluded from confirmatory analysis: SOS, EOS, LOS, and any exploratory response.

## Spatial design

- **Original sample:** 25 independent 5 km blocks across Shanghai, each with one tree and one grass candidate patch (50 patches total).
- **Independent replication:** 15 new 5 km blocks defined before outcome extraction, stratified across low/medium/high built-fraction and separated by ≥6 km from all other blocks; 30 candidate patches (one tree, one grass per block). After historical validation, 12 patches in 9 blocks remained.
- All patches were validated against historical imagery, blinded to NDVI outcomes.

## Exposure and covariates

- Primary exposure: patch-level **built fraction within 1,000 m**.
- 90 m built fraction is sensitivity-only.
- Crop context: crop fraction within 1,000 m.
- Block structure: independent 5 km blocks used for cluster-robust inference.

## NDVI extraction

- Sentinel-2 L2A is the reference sensor (cloud mask via SCL classes 4–7).
- Landsat C2 L2 is cross-calibrated to Sentinel-2 and used as calibration support only.
- Fixed temporal QC before extraction: ≥15 valid observations per patch-year, ≥3 valid July–August observations, maximum active-season gap ≤45 days; only real observations, no interpolation.

## Statistical model

For each endpoint:

```
endpoint ~ built_fraction_1000m + C(vegetation_type) + C(year) + crop_context_1000m
```

- Patch-year OLS.
- Standard errors clustered by independent 5 km block.
- 1,000-replicate block bootstrap.
- Leave-one-block-out refits.
- Standalone original-validation and standalone replication results are reported before any combined fit.

## Cross-sensor calibration

Three predeclared sensor strategies were evaluated:

- **A.** Sentinel-2 only.
- **B.** Sentinel-2 + pooled Landsat calibration.
- **C.** Sentinel-2 + platform-aware validated Landsat calibration.

Per platform group and temporal window, n, slope, intercept, RMSE, signed bias, vegetation-type differences, and annual stability are reported. A platform calibration is "usable" only if it meets minimum pair count, RMSE, bias, slope, type-bias, and year-coverage criteria; Landsat 7 is always modeled separately and never silently pooled. Outliers remain in the audit table (`outputs/tables/sensor_matched_pairs.csv`).

## Patch validation

All 50 original patches were independently audited with historical imagery. Confirmatory inclusion required persistent vegetation with HIGH or MEDIUM confidence and a clear tree/grass type. Results are in `outputs/tables/patch_validation.csv`.

## Independent replication

The external block design and validation were defined before NDVI extraction. The replication result is reported standalone, then combined with the original sample only afterward.

## Pipeline scripts (`scripts/`)

| Script | Role |
|---|---|
| `extract_sentinel2.py` | Sentinel-2 extraction for the study |
| `harmonize_sentinel2.py` | Sentinel-2 baseline harmonization |
| `extract_landsat.py` | Landsat extraction and temporal gap fill |
| `audit_landcover.py` | Land-cover / conversion audit |
| `extract_thermal_lst.py` | Landsat LST extraction |
| `analyze_original_sample.py` | Primary models and block-bootstrap/LOO |
| `make_design_figures.py` | Exploratory study figures |
| `audit_sensor_calibration.py` | Cross-sensor calibration audit |
| `validate_patches.py` | Patch ground-truth validation |
| `design_replication.py` | Replication block/patch design |
| `extract_replication.py` | Replication NDVI extraction |
| `validate_replication.py` | Replication patch validation |
| `analyze_replication.py` | Standalone replication models |
| `analyze_validated_original.py` | Validated original-sample models |
| `analyze_combined_sample.py` | Combined original + replication models |
| `analyze_thermal.py` | Secondary thermal consistency check |
| `thermal_local_fallback.py` | Local thermal fallback |
| `make_figures.py` | Final publication figures |
| `export_gis.py` | GeoJSON design assets |
| `historical_imagery_validation_assets.py` | Historical-imagery validation contact sheets |

Configuration is in `config/final_analysis.json` (random seed 42; analysis CRS EPSG:32651; display CRS EPSG:4326).
