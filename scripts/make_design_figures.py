"""Generate the study figure set (13 figures)."""
from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "assets" / "figures"
DATA = ROOT / "data" / "processed"
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "figure.dpi": 150,
    "savefig.dpi": 320, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
})
BLUE, ORANGE, GREEN, RED, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6B7280"
TYPE_COLOR = {"tree": GREEN, "grass": ORANGE}


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_zero(ax):
    ax.axhline(0, color="#333333", lw=0.8, ls="--", zorder=0)


def draw_geojson_polygons(ax, path, facecolor="#F4F1EA", edgecolor="#A8A29E"):
    """Draw Polygon/MultiPolygon boundaries without a geopandas dependency."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for feature in data["features"]:
        geom = feature["geometry"]
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for polygon in polygons:
            ring = np.asarray(polygon[0])
            ax.fill(ring[:, 0], ring[:, 1], facecolor=facecolor, edgecolor=edgecolor, linewidth=.6, zorder=0)


patches = pd.read_csv(OUT / "sampled_patches.csv")
blocks = pd.read_csv(OUT / "sampled_blocks.csv")
metrics = pd.read_csv(OUT / "seasonal_metrics.csv")
primary = metrics[(metrics.series == "S2+Landsat-fill") & metrics.included_primary.astype(bool)].copy()

# 1. Study design and urbanization gradient
fig, ax = plt.subplots(figsize=(7.1, 6.0))
draw_geojson_polygons(ax, ROOT / "data" / "raw" / "shanghai_boundary.geojson")
sc = ax.scatter(blocks.centroid_lon, blocks.centroid_lat, c=blocks.built_fraction,
                cmap="viridis", s=58, edgecolor="white", linewidth=0.65, zorder=3)
cb = fig.colorbar(sc, ax=ax, shrink=.72, pad=.02)
cb.set_label("Built fraction within 5 km block")
ax.set(title=" study design: 25 independent blocks", xlabel="Longitude (°E)", ylabel="Latitude (°N)")
ax.text(.01, .01, "Projection: WGS 84 (EPSG:4326)  |  Blocks fixed before endpoint analysis",
        transform=ax.transAxes, fontsize=7, color=GREY)
ax.set_aspect("equal", adjustable="datalim")
save(fig, "fig01_study_design_urban_gradient")

# 2. Patch distribution and type
fig, ax = plt.subplots(figsize=(7.1, 6.0))
draw_geojson_polygons(ax, ROOT / "data" / "raw" / "shanghai_boundary.geojson", "#F7F7F7", "#BDBDBD")
for t, marker in [("tree", "o"), ("grass", "^")]:
    d = patches[patches.vegetation_type == t]
    ax.scatter(d.centroid_lon, d.centroid_lat, c=TYPE_COLOR[t], marker=marker, s=48,
               edgecolor="white", linewidth=.7, label=f"{t.title()} patches (n={len(d)})", zorder=3)
ax.legend(frameon=False, loc="lower right")
ax.set(title="Fixed tree and grass patches", xlabel="Longitude (°E)", ylabel="Latitude (°N)")
ax.text(.01, .01, "50 patches; two per retained block  |  WGS 84", transform=ax.transAxes, fontsize=7, color=GREY)
ax.set_aspect("equal", adjustable="datalim")
save(fig, "fig02_patch_distribution_type")

# 3. QC retention and observation coverage
ret = pd.read_csv(OUT / "qc_retention_by_year.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), sharex=True)
for series, c, marker in [("S2-only", BLUE, "o"), ("S2+Landsat-fill", ORANGE, "s")]:
    d = ret[ret.series == series]
    axes[0].plot(d.year, d.temporal_qc_retained, marker=marker, color=c, label=series)
    axes[1].plot(d.year, d.primary_retained, marker=marker, color=c, label=series)
for ax, ttl in zip(axes, ["Temporal QC retained", "Final primary retained"]):
    ax.set(title=ttl, xlabel="Year", ylabel="Patch-years")
    ax.set_xticks(range(2017, 2026, 2)); ax.set_ylim(0, 52)
axes[0].legend(frameon=False)
fig.suptitle("QC retention improved with real Landsat temporal gap filling", y=1.02)
save(fig, "fig03_qc_retention_by_year")

# 4–5. Endpoint scatterplots with descriptive fit
for n, endpoint, label in [(4, "peak_NDVI", "Peak NDVI"), (5, "summer_NDVI", "Summer NDVI")]:
    fig, ax = plt.subplots(figsize=(4.7, 3.8))
    for t in ["tree", "grass"]:
        d = primary[primary.vegetation_type == t]
        ax.scatter(d.built_fraction_1000m, d[endpoint], s=19, alpha=.62,
                   c=TYPE_COLOR[t], label=t.title(), edgecolor="none")
    x = primary.built_fraction_1000m.to_numpy(); y = primary[endpoint].to_numpy()
    slope, intercept, *_ = linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, intercept + slope * xx, color="#222222", lw=1.4, label="Descriptive pooled fit")
    ax.set(xlabel="Built fraction (1,000 m)", ylabel=label,
           title=f"Urbanization gradient and {label.lower()}")
    ax.legend(frameon=False)
    ax.text(.02, .02, "Points are retained patch-years; confirmatory estimates adjust for type, year, and crop context.",
            transform=ax.transAxes, fontsize=6.8, color=GREY)
    save(fig, f"fig{n:02d}_built_vs_{endpoint.lower()}")

# 6–7. Annual forest plots
annual = pd.read_csv(OUT / "year_specific_results.csv")
for n, endpoint, label in [(6, "peak_NDVI", "Peak NDVI"), (7, "summer_NDVI", "Summer NDVI")]:
    d = annual[annual.endpoint == endpoint].sort_values("year")
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.errorbar(d.year, d.estimate, yerr=[d.estimate-d.ci_low, d.ci_high-d.estimate],
                fmt="o", color=BLUE, ecolor=BLUE, capsize=2.5, lw=1)
    add_zero(ax)
    ax.set(xticks=d.year, xlabel="Year", ylabel="Built-fraction coefficient",
           title=f"Annual urbanization effects on {label.lower()}")
    ax.tick_params(axis="x", rotation=45)
    save(fig, f"fig{n:02d}_annual_{endpoint.lower()}_effects")

# 8. Tree–grass estimates
tg = pd.read_csv(OUT / "tree_grass_results.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), sharex=True)
for ax, endpoint, ttl in zip(axes, ["peak_NDVI", "summer_NDVI"], ["Peak NDVI", "Summer NDVI"]):
    r = tg[tg.endpoint == endpoint].iloc[0]
    d = pd.DataFrame([
        {"type":"grass", "estimate":r.grass_slope, "ci_low":r.grass_ci_low, "ci_high":r.grass_ci_high},
        {"type":"tree", "estimate":r.tree_slope, "ci_low":r.tree_ci_low, "ci_high":r.tree_ci_high},
    ])
    y = np.arange(len(d))
    ax.errorbar(d.estimate, y, xerr=[d.estimate-d.ci_low, d.ci_high-d.estimate], fmt="none", ecolor=GREY, capsize=3)
    for yi, (_, r) in zip(y, d.iterrows()): ax.scatter(r.estimate, yi, c=TYPE_COLOR[r["type"]], s=42)
    ax.axvline(0, color="#333", lw=.8, ls="--")
    ax.set(yticks=y, yticklabels=d.type.str.title(), xlabel="Built-fraction coefficient", title=ttl)
fig.suptitle("Vegetation-type-specific urbanization response", y=1.02)
save(fig, "fig08_tree_grass_response")

# 9. Built fraction vs annual block LST
thermal = pd.read_csv(OUT / "thermal_metrics.csv")
fig, ax = plt.subplots(figsize=(4.8, 3.8))
sc = ax.scatter(thermal.block_built_fraction, thermal.summer_median_lst_c, c=thermal.year,
                cmap="viridis", s=23, alpha=.72, edgecolor="none")
x, y = thermal.block_built_fraction, thermal.summer_median_lst_c
slope, intercept, *_ = linregress(x, y)
xx = np.linspace(x.min(), x.max(), 100); ax.plot(xx, intercept+slope*xx, color="#222", lw=1.4)
fig.colorbar(sc, ax=ax, pad=.02, label="Year")
ax.set(xlabel="Built fraction (1,000 m)", ylabel="Summer LST (°C)", title="Urbanization and summer land-surface temperature")
save(fig, "fig09_built_vs_summer_lst")

# 10. LST vs summer NDVI
join = pd.read_csv(OUT / "patch_year_thermal_join.csv")
join = join[join.included_primary.astype(bool) & join.summer_median_lst_c.notna()]
fig, ax = plt.subplots(figsize=(4.8, 3.8))
for t in ["tree", "grass"]:
    d = join[join.vegetation_type == t]
    ax.scatter(d.summer_median_lst_c, d.summer_NDVI, c=TYPE_COLOR[t], s=20, alpha=.65, label=t.title(), edgecolor="none")
x, y = join.summer_median_lst_c, join.summer_NDVI
slope, intercept, *_ = linregress(x, y); xx=np.linspace(x.min(), x.max(), 100)
ax.plot(xx, intercept+slope*xx, color="#222", lw=1.4)
ax.set(xlabel="Summer LST (°C)", ylabel="Summer NDVI", title="Thermal exposure and summer greenness")
ax.legend(frameon=False)
save(fig, "fig10_lst_vs_summer_ndvi")

# 11. Leave-one-block-out robustness
loo = pd.read_csv(OUT / "leave_one_block_out.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.5), sharey=True)
order = sorted(loo.omitted_block.unique())
for ax, endpoint, ttl in zip(axes, ["peak_NDVI", "summer_NDVI"], ["Peak NDVI", "Summer NDVI"]):
    d = loo[loo.endpoint == endpoint].set_index("omitted_block").loc[order]
    y=np.arange(len(d)); ax.errorbar(d.estimate, y, xerr=[d.estimate-d.ci_low,d.ci_high-d.estimate], fmt="o", ms=3, color=BLUE, capsize=1.5)
    ax.axvline(0, color="#333", lw=.8, ls="--"); ax.set(title=ttl, xlabel="Coefficient", yticks=y, yticklabels=order)
fig.suptitle("Leave-one-block-out estimates retain a negative sign", y=.98)
save(fig, "fig11_leave_one_block_out")

# 12. Land-cover and scale sensitivity
sens = pd.read_csv(OUT / "landcover_sensitivity.csv")
labels = {"HIGH_MEDIUM_primary":"Primary", "HIGH_only":"High confidence only",
          "exclude_crop_context_ge_0.5":"Crop context < 0.5", "local_90m":"90 m built fraction",
          "S2_only_HIGH_MEDIUM":"Sentinel-2 only"}
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.7), sharey=True)
for ax, endpoint, ttl in zip(axes, ["peak_NDVI", "summer_NDVI"], ["Peak NDVI", "Summer NDVI"]):
    d=sens[sens.endpoint==endpoint].copy(); d["label"]=d.analysis.map(labels); y=np.arange(len(d))
    ax.errorbar(d.estimate,y,xerr=[d.estimate-d.ci_low,d.ci_high-d.estimate],fmt="o",color=BLUE,capsize=2)
    ax.axvline(0,color="#333",lw=.8,ls="--"); ax.set(title=ttl,xlabel="Coefficient",yticks=y,yticklabels=d.label)
fig.suptitle("Sensitivity analyses reveal sensor and confidence dependence", y=1.01)
save(fig, "fig12_landcover_scale_sensor_sensitivity")

# 13. Representative seasonal curves by type and urban context
ts = pd.read_csv(DATA / "combined_patch_timeseries.csv", parse_dates=["date"])
ts = ts.merge(patches[["patch_id","built_fraction_1000m"]], on="patch_id", how="left")
ts["context"] = pd.cut(ts.built_fraction_1000m, [-np.inf,.2,.5,np.inf], labels=["low","medium","high"])
ts["doy"] = ts.date.dt.dayofyear
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), sharey=True)
for ax, t in zip(axes, ["tree","grass"]):
    for context, color in [("low",BLUE),("high",RED)]:
        d=ts[(ts.vegetation_type==t)&(ts.context==context)]
        b=(d.assign(bin=(d.doy//15)*15).groupby("bin").ndvi.median().reset_index())
        ax.plot(b.bin,b.ndvi,color=color,lw=1.8,label=f"{context.title()} urban context")
    ax.axvspan(152,243,color="#FDE68A",alpha=.23,lw=0)
    ax.set(title=t.title(),xlabel="Day of year",ylabel="Median NDVI",xlim=(55,330),ylim=(0,0.85))
axes[0].legend(frameon=False)
fig.suptitle("Seasonal greenness trajectories (15-day medians, all years)", y=1.02)
save(fig, "fig13_representative_seasonal_curves")

print(f"Generated 13 PNG and 13 PDF figures in {FIG}")
