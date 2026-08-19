# Limitations

A frank account of what the evidence does and does not support. These limits are carried into the README and should accompany any reuse of the results.

## Design and inference

- **Observational, not causal.** The study relates an exposure (surrounding built fraction) to an outcome (vegetation greenness) across space and time. There is no randomization or intervention, so the results are associations, not causal effects. The README and reports avoid causal language deliberately.
- **Block-bootstrap uncertainty remains material.** The combined crop-adjusted summer coefficient is −0.114 (95% CI −0.228 to 0.001; p = 0.051; block bootstrap −0.249 to 0.074). The adjusted cluster interval touches zero and the bootstrap interval crosses zero, so the inferential threshold is not fully cleared despite consistent direction.

## Land cover and patch purity

- Patch typing and land-cover context rely on class-based products (ESA WorldCover, IO LULC) and visual historical validation. Peri-urban crop contamination is present and is controlled via the crop-context covariate, but residual misclassification is possible.
- Patch "purity" is class-based, not species-resolved.

## Cross-sensor calibration

- Calibration coefficients show annual instability. Landat 8/9 platform slopes range 0.739–1.151 (RMSE 0.104–0.124); Landsat 7 failed the slope criterion and was excluded from validated gap filling. Platform-aware calibration reduces pooling bias but does not resolve time-varying uncertainty.
- Sentinel-2-only peak NDVI reverses sign relative to calibrated strategies, which is why peak NDVI is downgraded to a secondary endpoint.
- The reported `bias` is a fitted residual bias (near zero by construction with an intercept); raw sensor differences remain traceable in `outputs/tables/sensor_matched_pairs.csv` and should be summarized explicitly in any manuscript supplement.

## Agricultural and crop context

- Crop adjustment materially attenuates estimates (combined crop-adjusted summer −0.114 vs no-crop −0.182). Rural/agricultural context therefore matters, and the direction is preserved rather than eliminated by it.

## Independent replication imbalance

- The replication retained only **one** low-urbanization block after historical validation, because agricultural, aquaculture, greenhouse, and mixed-land-cover contamination removed most low-urbanization candidates. Its block-bootstrap interval crossed zero, and its magnitude is not directly comparable to the original sample. The replication is best read as a directional consistency check, not an independent effect-size estimate.

## Thermal secondary analysis

- Thermal evidence (built → summer LST positive; summer LST → summer NDVI negative) is directionally coherent but uncertain (p = 0.077 and p = 0.579 respectively) and is restricted to the originally validated blocks. It is supporting secondary evidence, not a confirmed mechanism.

## Provenance and tooling

- Historical-imagery Wayback version dates are publication dates, not guaranteed acquisition dates.
- The editable QGIS project (`.qgz`) was not re-QA-verified in the final pass, so GeoJSON and static maps are delivered without a newly validated editable project file.
