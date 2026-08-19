"""
plot_main_figures.py — Redraw priority publication figures from committed data.

Regenerates figures 01, 06, 08, 09, 11, 12 using only committed tables
(outputs/tables/*.csv) and committed GIS assets (gis/*.geojson).

Figure 07 (replication summer scatter) is intentionally NOT regenerated.
Its per-point NDVI table (replication_primary_sample.csv) was never committed
to the repository, so the scatter is not reproducible from the published
outputs. The existing PNG is retained in assets/figures/ as a static artifact,
but it is not referenced from README; the independent replication evidence is
carried by Figure 08 and the committed replication tables.

Data sources per figure:
  Fig 01 — gis/shanghai_boundary.geojson (basemap, EPSG:4326 → plotted in EPSG:32651),
           gis/validated_original_patches.geojson,
           gis/validated_replication_patches.geojson,
           gis/replication_blocks.geojson
  Fig 06 — outputs/tables/validated_primary_sample.csv
  Fig 08 — outputs/tables/original_vs_replication.csv
  Fig 09 — outputs/tables/combined_primary_models.csv (summer_NDVI rows; crop-adjusted + no-crop-context)
  Fig 11 — outputs/tables/combined_leave_one_block_out.csv + combined_primary_models.csv (full-sample estimate)
  Fig 12 — outputs/tables/thermal_validation.csv

All scientific values (coefficients, CIs, p-values, sample sizes, the full-sample
estimate) are read directly from committed CSVs and are not hard-coded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pyproj import Transformer

# ── Repository paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
GIS = ROOT / "gis"
FIGS = ROOT / "assets" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Consistent academic style ─────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.labelsize": 10.5,
        "axes.labelweight": "normal",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.12,
    }
)

# ── Color palette (colorblind-friendly, restrained) ────────────────────────
C_PRIMARY = "#2B6CB0"       # blue   — main / original relationship
C_REPLICATION = "#D97706"   # amber  — replication / secondary category
C_SECONDARY = "#7A7A7A"     # gray   — secondary endpoint (Peak NDVI) / captions
C_FIT = "#C05621"          # burnt-orange — descriptive trend line
C_ZERO = "#444444"          # dark gray — zero reference line / scale bar
C_GRID = "#EDEDED"          # near-white — subtle grid
C_BLOCK_EDGE = "#8C949C"    # light gray — block boundary

# ── Projection: display CRS EPSG:4326 → study/analysis CRS EPSG:32651 (UTM 51N)
_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)


def _utm_km(lon: float, lat: float) -> tuple[float, float]:
    """Project lon/lat to UTM zone 51N and return kilometres Easting/Northing."""
    e, n = _UTM.transform(float(lon), float(lat))
    return e / 1000.0, n / 1000.0


# ═══════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════


def _save(fig_num: int, name: str, tight: bool = False) -> None:
    """Save current figure as PNG and PDF."""
    png_path = FIGS / f"figure_{fig_num:02d}_{name}.png"
    pdf_path = FIGS / f"figure_{fig_num:02d}_{name}.pdf"
    plt.savefig(png_path, bbox_inches="tight" if tight else None)
    plt.savefig(pdf_path, bbox_inches="tight" if tight else None)
    plt.close()
    print(f"  saved {png_path.name}  ({png_path.stat().st_size // 1024} KB)")


def _north_arrow(ax: plt.Axes, x: float = 0.96, y: float = 0.96, size: float = 14) -> None:
    """Draw a minimal north-arrow annotation in axes coordinates."""
    ax.text(
        x, y, "\u2191 N",
        transform=ax.transAxes,
        fontsize=size, fontweight="bold",
        color=C_ZERO, ha="center", va="center",
        zorder=99,
    )


def _load_geojson(path: Path) -> list[dict]:
    """Load GeoJSON and return features list."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["features"]


def _polygon_coords(feature: dict) -> list[tuple[float, float]]:
    """Extract exterior ring coordinates from a Polygon feature (lon/lat)."""
    geom = feature["geometry"]
    if geom["type"] == "Polygon":
        return [(c[0], c[1]) for c in geom["coordinates"][0]]
    raise ValueError(f"Unsupported geometry type: {geom['type']}")


def _scale_bar(ax: plt.Axes, km: float = 5.0) -> None:
    """Draw a small, clean scale bar of length `km` in the lower-left of the map."""
    x0 = ax.get_xlim()[0] + 1.2
    y0 = ax.get_ylim()[0] + 1.2
    ax.plot([x0, x0 + km], [y0, y0], color=C_ZERO, lw=1.4, zorder=7, clip_on=False)
    ax.plot([x0, x0], [y0 - 0.25, y0 + 0.25], color=C_ZERO, lw=1.4, zorder=7, clip_on=False)
    ax.plot([x0 + km, x0 + km], [y0 - 0.25, y0 + 0.25], color=C_ZERO, lw=1.4, zorder=7, clip_on=False)
    ax.text(x0 + km / 2, y0 + 0.4, f"{km:.0f} km", color=C_ZERO,
            fontsize=7.5, ha="center", va="bottom", zorder=7)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 01 — Patch map (projected to study CRS EPSG:32651)
# ═══════════════════════════════════════════════════════════════════════════


def fig_01_patch_map() -> None:
    print("Generating Figure 01: Patch map (EPSG:32651) ...")

    orig_feats = _load_geojson(GIS / "validated_original_patches.geojson")          # 50
    repl_feats = _load_geojson(GIS / "validated_replication_patches.geojson")      # 30
    block_feats = _load_geojson(GIS / "replication_blocks.geojson")                # 15

    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    # --- Shanghai basemap (subtle land context), projected to UTM km ---
    d = json.load(open(GIS / "shanghai_boundary.geojson", encoding="utf-8"))
    for feat in d["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = poly[0]
            coords = [_utm_km(c[0], c[1]) for c in ring]
            ax.add_patch(mpatches.Polygon(
                coords, closed=True,
                facecolor="#F5F5F4", edgecolor="#DCDCDC",
                linewidth=0.4, zorder=0,
            ))

    # --- Replication block boundaries (clear outlines over basemap) ---
    for bf in block_feats:
        try:
            coords = _polygon_coords(bf)
            utm = [_utm_km(x, y) for (x, y) in coords]
            ax.add_patch(mpatches.Polygon(
                utm, closed=True,
                facecolor="none", edgecolor="#8C949C",
                linewidth=0.8, zorder=1,
            ))
        except ValueError:
            pass  # skip non-polygon geometries silently

    # --- Original patches (circles) ---
    for feat in orig_feats:
        p = feat["properties"]
        x, y = _utm_km(p["centroid_lon"], p["centroid_lat"])
        inc = str(p.get("include_confirmatory", "")).lower() == "true"
        alpha = 0.90 if inc else 0.32
        size = 54 if inc else 36
        ax.scatter(x, y, marker="o", s=size,
                   c=C_PRIMARY, alpha=alpha,
                   edgecolors="white", linewidths=0.45, zorder=3)

    # --- Replication patches (triangles) ---
    for feat in repl_feats:
        p = feat["properties"]
        x, y = _utm_km(p["centroid_lon"], p["centroid_lat"])
        inc = str(p.get("include_replication_confirmatory", "")).lower() == "true"
        alpha = 0.90 if inc else 0.32
        size = 62 if inc else 42
        ax.scatter(x, y, marker="^", s=size,
                   c=C_REPLICATION, alpha=alpha,
                   edgecolors="white", linewidths=0.4, zorder=3)

    # Map extent — frame all patch centroids with a little padding
    xs = [f["properties"]["centroid_lon"] for f in orig_feats + repl_feats]
    ys = [f["properties"]["centroid_lat"] for f in orig_feats + repl_feats]
    (ex0, ey0) = _utm_km(min(xs), min(ys))
    (ex1, ey1) = _utm_km(max(xs), max(ys))
    pad_x = (ex1 - ex0) * 0.08
    pad_y = (ey1 - ey0) * 0.08
    ax.set_xlim(ex0 - pad_x, ex1 + pad_x)
    ax.set_ylim(ey0 - pad_y, ey1 + pad_y)
    ax.set_aspect("equal")

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_title("Validated vegetation patches across Shanghai", loc="left", pad=10)

    # North arrow (top-left, clear of markers and outside-legend space)
    _north_arrow(ax, x=0.04, y=0.95)

    # Scale bar (5 km) — clean, lower-left
    _scale_bar(ax, km=5.0)

    # Legend placed OUTSIDE the plot (right side) so it never covers markers
    legend_handles = [
        mpatches.Patch(facecolor=C_PRIMARY, edgecolor="white",
                       label=f"Original patch (n={len(orig_feats)})"),
        mpatches.Patch(facecolor=C_REPLICATION, edgecolor="white",
                       label=f"Replication patch (n={len(repl_feats)})"),
        mpatches.Patch(facecolor="none", edgecolor="#8C949C", linewidth=1.2,
                       label=f"Replication block (n={len(block_feats)})"),
    ]
    ax.legend(handles=legend_handles, frameon=True, fancybox=False,
              fontsize=9, edgecolor="#CCCCCC", handlelength=1.4,
              loc="center left", bbox_to_anchor=(1.01, 0.5), borderpad=0.7,
              title="Patch type", title_fontsize=9.5)

    # Opacity note as a caption BELOW the x-axis label (clear of it)
    ax.text(0.5, -0.17,
            "Opaque markers entered the analysis; faded markers were excluded.",
            transform=ax.transAxes, fontsize=7.5, color=C_SECONDARY,
            style="italic", ha="center", va="top")

    _save(1, "validated_patch_map", tight=True)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 06 — Original validated sample: built fraction vs summer NDVI
# ═══════════════════════════════════════════════════════════════════════════


def fig_06_original_scatter() -> None:
    print("Generating Figure 06: Original summer scatter ...")

    df = pd.read_csv(TABLES / "validated_primary_sample.csv")
    sub = df[
        (df.sensor_strategy == "C_platform_aware")
        & (df.include_confirmatory.astype(str).str.lower() == "true")
    ].copy()

    n_py = len(sub)
    n_pat = sub.patch_id.nunique()
    n_blk = sub.block_id.nunique()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.subplots_adjust(bottom=0.20)

    ax.scatter(sub.built_fraction_1000m, sub.summer_NDVI,
               c=C_PRIMARY, s=38, alpha=0.62,
               edgecolors="white", linewidths=0.35, zorder=3)

    # Visual descriptive trend line (np.polyfit degree-1). This is an UNadjusted
    # descriptive linear fit; it is NOT the inferential model coefficient.
    z = np.polyfit(sub.built_fraction_1000m, sub.summer_NDVI, 1)
    xx = np.linspace(sub.built_fraction_1000m.min(), sub.built_fraction_1000m.max(), 200)
    ax.plot(xx, np.polyval(z, xx), color=C_FIT, linewidth=1.8, zorder=4)

    ax.set_xlabel("Built fraction within 1,000 m")
    ax.set_ylabel("Summer NDVI")
    ax.set_title("Original validated sample", loc="left", pad=8)

    # Sample annotation
    ax.text(0.97, 0.04, f"n = {n_py} patch-years \u00b7 {n_pat} patches \u00b7 {n_blk} blocks",
            transform=ax.transAxes, fontsize=8, color=C_SECONDARY,
            ha="right", va="bottom")

    # Descriptive-trend disclaimer (below the x-axis label)
    ax.text(0.5, -0.16,
            "Line shows an unadjusted descriptive linear trend.",
            transform=ax.transAxes, fontsize=7.3, color=C_SECONDARY,
            style="italic", ha="center", va="top")

    # Subtle horizontal grid
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    _save(6, "original_summer_scatter")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 08 — Original versus independent replication effect comparison
# ═══════════════════════════════════════════════════════════════════════════


def fig_08_comparison() -> None:
    print("Generating Figure 08: Original vs replication comparison ...")

    df = pd.read_csv(TABLES / "original_vs_replication.csv").copy()

    # Build display order: Summer primary (top), Peak secondary (bottom)
    endpoint_order = {"summer_NDVI": 0, "peak_NDVI": 1}
    sample_order = {"independent_replication": 0, "original_validated": 1}
    df["_ep_rank"] = df["endpoint"].map(endpoint_order)
    df["_sp_rank"] = df["sample"].map(sample_order)
    df = df.sort_values(["_ep_rank", "_sp_rank"]).reset_index(drop=True)

    n_rows = len(df)

    # X-limits set before plotting so coefficient labels can sit clear of the whiskers
    x_left = min(0.0, df.ci_low.min()) - 0.04
    x_right = max(0.0, df.ci_high.max()) + 0.12
    x_span = x_right - x_left

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.set_xlim(x_left, x_right)

    for i, row in df.iterrows():
        ep_pri = (row.endpoint == "summer_NDVI")
        color = C_PRIMARY if row["sample"] == "original_validated" else C_REPLICATION
        ms = 9 if ep_pri else 6.5
        lw_eb = 1.25 if ep_pri else 0.95
        ec = color if ep_pri else C_SECONDARY
        mc = color if ep_pri else C_SECONDARY

        lo = row.estimate - row.ci_low
        hi = row.ci_high - row.estimate
        ax.errorbar(row.estimate, i,
                    xerr=[[lo], [hi]],
                    fmt="o", ms=ms, color=mc,
                    ecolor=ec, elinewidth=lw_eb, capsize=3,
                    zorder=5 if ep_pri else 4)

        # Coefficient label placed to the RIGHT of the CI cap, so it never
        # sits on top of the error bar.
        ax.annotate(f"{row.estimate:+.3f}",
                    xy=(row.ci_high, i),
                    xytext=(x_span * 0.02, 0), textcoords="offset points",
                    fontsize=8.5 if ep_pri else 7.5,
                    fontweight="bold" if ep_pri else "normal",
                    color=mc, va="center", ha="left")

    # Zero line
    ax.axvline(0, color=C_ZERO, linestyle="--", linewidth=0.85, zorder=1)

    # Y-axis labels
    ylabels = []
    for _, r in df.iterrows():
        sp_name = "Original" if r["sample"] == "original_validated" else "Replication"
        ylabels.append(f"{sp_name}")
    yticks_pos = list(range(n_rows))
    ax.set_yticks(yticks_pos)
    ax.set_yticklabels(ylabels)

    # Add endpoint group labels as text annotations on the left
    summer_idx = df[df.endpoint == "summer_NDVI"].index.tolist()
    peak_idx = df[df.endpoint == "peak_NDVI"].index.tolist()
    if summer_idx:
        mid_s = (min(summer_idx) + max(summer_idx)) / 2
        ax.text(-0.18, mid_s, "Summer NDVI", transform=ax.get_yaxis_transform(),
                fontsize=10, fontweight="bold", color=C_PRIMARY, va="center", ha="right")
    if peak_idx:
        mid_p = (min(peak_idx) + max(peak_idx)) / 2
        ax.text(-0.18, mid_p, "Peak NDVI", transform=ax.get_yaxis_transform(),
                fontsize=10, fontweight="normal", color=C_SECONDARY, va="center", ha="right")

    ax.set_xlabel("Built-fraction coefficient (95% cluster CI)")

    # Note about replication magnitude — placed OUTSIDE the plot (below) so it
    # never covers the data or the coefficient labels.
    ax.text(0.5, -0.20,
            "Note: replication magnitude is not directly comparable "
            "(sample imbalance; low-urbanization stratum collapsed).",
            transform=ax.transAxes, fontsize=7.3, color=C_SECONDARY,
            ha="center", va="top", style="italic")

    _save(8, "original_vs_replication", tight=True)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 09 — Combined final summer NDVI effect (primary + one sensitivity)
# ═══════════════════════════════════════════════════════════════════════════


def fig_09_combined_summer() -> None:
    print("Generating Figure 09: Combined summer NDVI effect ...")

    df = pd.read_csv(TABLES / "combined_primary_models.csv")
    summer = df[df.endpoint == "summer_NDVI"].copy().reset_index(drop=True)

    # Only directly interpretable sensitivity models are shown as comparable
    # coefficients: crop-adjusted (primary) and no-crop-context. The vegetation
    # interaction model's built-fraction main effect is the slope for the
    # reference vegetation category, not the study-wide effect, so it is omitted
    # from this plot (its results remain in combined_primary_models.csv).
    model_order = {"crop_adjusted": 0, "no_crop_context": 1}
    summer = summer[summer.model.isin(model_order)].copy()
    summer["_rank"] = summer.model.map(model_order)
    summer = summer.sort_values("_rank").reset_index(drop=True)

    n_rows = len(summer)

    # X-limits set before plotting; right margin leaves room for the combined
    # contrast + p-value annotation, left margin for the coefficient labels.
    x_left = min(0.0, summer.ci_low.min()) - 0.04
    x_right = max(0.0, summer.ci_high.max()) + 0.18
    x_span = x_right - x_left

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.set_xlim(x_left, x_right)

    for i, (_, row) in enumerate(summer.iterrows()):
        is_primary = (row.model == "crop_adjusted")
        mc = C_PRIMARY if is_primary else C_SECONDARY
        ec = C_PRIMARY if is_primary else C_SECONDARY
        ms = 10 if is_primary else 7
        lwe = 1.3 if is_primary else 1.0
        z = 5 if is_primary else 4

        lo = row.estimate - row.ci_low
        hi = row.ci_high - row.estimate
        ax.errorbar(row.estimate, i,
                    xerr=[[lo], [hi]],
                    fmt="o", ms=ms, color=mc,
                    ecolor=ec, elinewidth=lwe, capsize=3.5,
                    zorder=z)

        # Coefficient label placed LEFT of the CI cap (clear of the error bar).
        ax.annotate(f"{row.estimate:.3f}",
                    xy=(row.ci_low, i),
                    xytext=(-x_span * 0.035, 0), textcoords="offset points",
                    fontsize=9 if is_primary else 8,
                    fontweight="bold" if is_primary else "normal",
                    color=mc, va="center", ha="right")

    # Primary model: contrast + p-value combined into ONE annotation to the RIGHT
    # of the CI cap, so the two pieces of text never collide with each other.
    prim = summer[summer.model == "crop_adjusted"].iloc[0]
    parts = []
    if pd.notna(prim.contrast_75pct_vs_5pct_built):
        parts.append(f"5%\u219275% built \u2248 {prim.contrast_75pct_vs_5pct_built:.3f}")
    if pd.notna(prim.p_value):
        pv = f"p = {prim.p_value:.3f}" if prim.p_value >= 0.001 else "p < 0.001"
        parts.append(pv)
    if parts:
        ax.annotate("  \u00b7  ".join(parts),
                    xy=(prim.ci_high, 0),
                    xytext=(x_span * 0.035, 0), textcoords="offset points",
                    fontsize=8.5, color=C_PRIMARY, fontweight="bold",
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=C_PRIMARY, lw=0.8))

    ax.axvline(0, color=C_ZERO, linestyle="--", linewidth=0.85, zorder=1)

    model_labels = {
        "crop_adjusted": "Crop-adjusted (primary)",
        "no_crop_context": "No crop context",
    }
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([model_labels[m] for m in summer.model])
    ax.set_xlabel("Built-fraction coefficient (95% cluster CI)")

    _save(9, "combined_summer_ndvi", tight=True)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 11 — Leave-one-block-out robustness (horizontal dot plot)
# ═══════════════════════════════════════════════════════════════════════════


def fig_11_loo() -> None:
    print("Generating Figure 11: Leave-one-block-out robustness ...")

    df = pd.read_csv(TABLES / "combined_leave_one_block_out.csv")

    summer_loo = df[(df.endpoint == "summer_NDVI") & (df.model == "crop_adjusted")].copy()
    peak_loo = df[(df.endpoint == "peak_NDVI") & (df.model == "crop_adjusted")].copy()

    blocks = summer_loo.left_out_block.values
    n_blocks = len(blocks)

    # Full-sample estimate read dynamically from the committed primary models
    # table (never hard-coded).
    models = pd.read_csv(TABLES / "combined_primary_models.csv")
    full_est = float(models.loc[
        (models["endpoint"] == "summer_NDVI") & (models["model"] == "crop_adjusted"),
        "estimate",
    ].iloc[0])

    fig, ax = plt.subplots(figsize=(7.6, 6.2))

    # Summer NDVI (primary) — main dots
    y_summer = np.arange(n_blocks, dtype=float)
    ax.scatter(summer_loo.estimate.values, y_summer,
               s=48, c=C_PRIMARY, zorder=5,
               edgecolors="white", linewidths=0.4, label="Summer NDVI")

    # Peak NDVI (secondary) — slightly offset upward so they don't overlap
    y_peak = y_summer + 0.28
    ax.scatter(peak_loo.estimate.values, y_peak,
               s=28, c=C_SECONDARY, zorder=4,
               edgecolors="white", linewidths=0.3, label="Peak NDVI")

    # Reference lines (no in-plot labels — kept clear of the dots)
    ax.axvline(0, color=C_ZERO, linestyle="--", linewidth=0.9, zorder=2)
    ax.axvline(full_est, color=C_PRIMARY, linestyle=":", linewidth=0.75,
               alpha=0.55, zorder=2)

    ax.set_yticks(y_summer)
    ax.set_yticklabels(blocks, fontsize=8.5)
    ax.set_xlabel("Coefficient (built fraction \u2192 endpoint)", labelpad=6)
    ax.set_ylabel("Left-out block", labelpad=6)
    # Extend downward so the full-sample label sits in a clear margin
    ax.set_ylim(-0.9, n_blocks + 0.4)

    # Full-sample estimate label — place it INSIDE the axes bottom margin,
    # well clear of the lowest data point (y=0) and the x-axis spine.
    ax.text(full_est, -0.65, f"Full-sample estimate ({full_est:.3f})",
            fontsize=7.8, color=C_PRIMARY, alpha=0.85,
            ha="center", va="top")

    # Legend OUTSIDE the plot (below), two entries side by side
    ax.legend(frameon=True, fancybox=False, fontsize=9.5,
              edgecolor="#CCCCCC", loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=2, borderpad=0.5,
              handletextpad=0.4, columnspacing=2.0)

    # Key message — placed well BELOW the legend using figure coords so it
    # never collides with the x-axis label or the legend itself.
    n_neg_s = int(summer_loo.estimate.lt(0).sum())
    n_neg_p = int(peak_loo.estimate.lt(0).sum())
    fig.text(0.5, 0.03,
             f"All {n_blocks}/{n_blocks} leave-one-block-out estimates remain negative "
             f"(Summer NDVI and Peak NDVI)",
             fontsize=9, color=C_PRIMARY, fontweight="bold",
             ha="center", va="top")

    # Subtle vertical grid at key positions
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.35, alpha=0.4)
    ax.set_axisbelow(True)

    _save(11, "combined_loo", tight=True)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 12 — Thermal consistency (two panels, separate x-axis scales)
# ═══════════════════════════════════════════════════════════════════════════


def _thermal_panel(ax: plt.Axes, row: pd.Series, title: str, xlabel: str) -> None:
    """Draw one thermal relation: estimate, 95% CI, zero reference, explicit units."""
    est = float(row.estimate)
    lo = est - float(row.ci_low)
    hi = float(row.ci_high) - est

    ax.errorbar(est, 0, xerr=[[lo], [hi]],
                fmt="o", ms=8, color=C_REPLICATION,
                ecolor=C_REPLICATION, elinewidth=1.3, capsize=4, zorder=5)

    ax.axvline(0, color=C_ZERO, linestyle="--", linewidth=0.85, zorder=1)

    span = abs(hi - lo)
    x_left = min(0.0, float(row.ci_low)) - span * 0.18
    x_right = max(0.0, float(row.ci_high)) + span * 0.18
    ax.set_xlim(x_left, x_right)

    # Value label placed to the RIGHT of the CI cap (clear of the error bar).
    ax.annotate(f"{est:+.4f}",
                xy=(float(row.ci_high), 0),
                xytext=(8, 0), textcoords="offset points",
                fontsize=8.5, color=C_REPLICATION, va="center", ha="left")

    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=10, pad=4)
    ax.set_xlabel(xlabel, fontsize=9)

    ax.xaxis.grid(True, color=C_GRID, linewidth=0.35, alpha=0.4)
    ax.set_axisbelow(True)


def fig_12_thermal() -> None:
    print("Generating Figure 12: Thermal consistency (separate scales) ...")

    df = pd.read_csv(TABLES / "thermal_validation.csv").copy()

    rowA = df.loc[df.relation == "built_fraction_to_summer_LST"].iloc[0]
    rowB = df.loc[df.relation == "summer_LST_to_summer_NDVI_adjusted_built"].iloc[0]

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(6.5, 4.4))

    _thermal_panel(axA, rowA,
                   "Built fraction \u2192 summer LST",
                   "Summer LST change (\u00b0C per full built-fraction unit)")
    _thermal_panel(axB, rowB,
                   "Summer LST \u2192 summer NDVI",
                   "Summer NDVI change per 1 \u00b0C")

    # Caption — supporting-only evidence; not mechanistic/mediation.
    fig.text(0.5, 0.02,
             "Directionally consistent supporting evidence; not a mechanistic or mediation test.",
             fontsize=7.4, color=C_SECONDARY, style="italic",
             ha="center", va="top")

    _save(12, "thermal_consistency", tight=True)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    print("=" * 60)
    print("Redrawing priority figures (committed-data reproducible)")
    print("=" * 60)

    fig_01_patch_map()
    fig_06_original_scatter()
    fig_08_comparison()
    fig_09_combined_summer()
    fig_11_loo()
    fig_12_thermal()

    print("-" * 60)
    n_png = len(list(FIGS.glob("*.png")))
    n_pdf = len(list(FIGS.glob("*.pdf")))
    print(f"Done. {FIGS.relative_to(ROOT)} now contains "
          f"{n_png} PNG and {n_pdf} PDF files.")
    print("\nNOTE: Figure 07 (replication summer scatter) was NOT regenerated.")
    print("Its per-point NDVI table (replication_primary_sample.csv) was never")
    print("committed to this repository, so the scatter is not reproducible from")
    print("the published outputs. The PNG is retained in assets/figures/ as a")
    print("static artifact but is not referenced from README; Figure 08 carries")
    print("the independent replication evidence instead.")


if __name__ == "__main__":
    main()
