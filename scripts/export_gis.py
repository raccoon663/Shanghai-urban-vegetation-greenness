"""Join validated spatial layers with patch-level result tables and export committed GeoJSON.

Source geometries are pipeline intermediates in outputs/; the patch-level
attribute tables that drive the join are committed final tables in
outputs/tables/. The exported validated layers are committed to gis/.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"                       # pipeline intermediates
TABLES = ROOT / "outputs" / "tables"      # committed final result tables
GIS = ROOT / "gis"                         # committed spatial layers
COMMITTED_TABLES = {"patch_validation.csv"}


def join_geo(src, table, key, out, origin):
    gj = json.loads((O / src).read_text(encoding="utf-8"))
    tpath = TABLES / table if table in COMMITTED_TABLES else O / table
    t = pd.read_csv(tpath)
    rows = {str(r[key]): r.where(pd.notna(r), None).to_dict() for _, r in t.iterrows()}
    features = []
    for f in gj["features"]:
        k = str(f["properties"][key])
        f["properties"].update(rows.get(k, {}))
        f["properties"]["sample_origin"] = origin
        features.append(f)
    (GIS / out).write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )


join_geo("sampled_patches.geojson", "patch_validation.csv", "patch_id", "validated_original_patches.geojson", "original")
join_geo("replication_patches.geojson", "replication_patch_validation.csv", "patch_id", "validated_replication_patches.geojson", "replication")
print("GIS validation layers written")
