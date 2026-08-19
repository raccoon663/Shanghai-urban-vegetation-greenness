# Data sources and processing

Detailed provenance for the inputs behind `outputs/tables/` and `assets/figures/`. See also `data/README.md` for the acquisition rationale and `docs/methods.md` for how they enter the pipeline.

## Satellite and land-cover collections

All collections were accessed through the Microsoft Planetary Computer STAC API (`https://planetarycomputer.microsoft.com/api/stac/v1`).

| Collection | ID | Role | Period |
|---|---|---|---|
| Sentinel-2 L2A | `sentinel-2-l2a` | Reference surface reflectance (10/20/60 m) | 2017–2025 |
| Landsat C2 L2 | `landsat-c2-l2` | Cross-calibration / gap fill (30 m) | 2017–2025 |
| ESA WorldCover | `esa-worldcover` | Built / tree / grass / crop / water context | 2020, 2021 |
| IO LULC annual | `io-lulc-annual-v02` | Annual land-cover, conversion checks | 2017–2023 |

## Processing parameters

- **Cloud masking:** Sentinel-2 SCL classes 4–7; Landsat QA bits 0–5 excluded.
- **Indices:** NDVI and EVI2; primary smoother Whittaker, sensitivity smoother Savitzky–Golay.
- **Active months:** March–November. **Summer months:** July–August. **Autumn months:** September–November.
- **Temporal candidate rule:** lowest scene cloud per MGRS tile and 10-day bin (Sentinel-2); lowest scene cloud per WRS path-row and 10-day bin (Landsat).
- **Sentinel-2 baseline offset:** harmonization start `2022-01-25`, reflectance offset `0.1` (per `config/final_analysis.json`).
- **Calibration window:** Landsat within ±3 days of a valid Sentinel-2 observation.
- **CRS:** analysis EPSG:32651 (UTM 51N); display EPSG:4326.
- **Block size:** 5 km. **Minimum new-block separation:** 6 km.

## Built-environment layers

- Built fraction computed from ESA WorldCover class 50 (built) within 1,000 m (primary) and 90 m (sensitivity) of each patch centroid.
- Crop fraction (class 40) within 1,000 m used as the crop-context covariate.
- Annual IO LULC used to flag probable conversions (built fraction ≥ 0.667 and vegetation fraction ≤ 0.333, persisted ≥ 2 years); converted patches are excluded from the conversion year onward.

## Historical imagery validation

Fixed Esri World Imagery Wayback versions representing early (2018), middle (2021), and late (2025) periods were inspected per patch. Note: Wayback version dates are release dates, not guaranteed acquisition dates — a residual temporal-provenance limitation.

## Administrative boundary (cartographic context only)

| Layer | Source | Role |
|---|---|---|
| Shanghai municipal boundary (16 districts) | Aliyun DataV.GeoAtlas (`310000_full.json`) — `https://geo.datav.aliyun.com/areas_v3/bound/310000_full.json` | AOI basemap for Figure 1 only |

- **Format:** GeoJSON (MultiPolygon, 16 district features), EPSG:4326; bundled as `gis/shanghai_boundary.geojson`.
- **Role in this repository:** static cartographic context for Figure 1 only; it is not used in any statistical computation, model fit, or area calculation.
- **License / reuse terms:** DataV.GeoAtlas boundaries are provided by Aliyun for mapping use; specific redistribution and commercial-use terms should be confirmed from the DataV.GeoAtlas attribution before any downstream reuse beyond this repository's display purpose.

## Reproducibility note

Raw scenes and intermediate rasters are not stored in this repository. To regenerate, acquire the collections above for the Shanghai AOI over 2017–2025, place them under the working `data/` layout expected by `scripts/`, and run the pipeline in `docs/methods.md` with `config/final_analysis.json`.
