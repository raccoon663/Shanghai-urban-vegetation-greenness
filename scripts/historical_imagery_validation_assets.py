"""Download fixed Esri Wayback tile vintages and create blinded patch triptychs/contact sheets."""
from pathlib import Path
from io import BytesIO
import math

import pandas as pd
import requests
import time
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
WORK = ROOT / "work" / "historical_imagery"
WORK.mkdir(parents=True, exist_ok=True)
import sys
REPLICATION = len(sys.argv) > 1 and sys.argv[1] == "replication"
PATCHES = pd.read_csv(TABLES / "replication_patches.csv" if REPLICATION else ROOT / "outputs" / "sampled_patches.csv")
if REPLICATION:
    PATCHES = PATCHES.rename(columns={"vegetation_type":"vegetation_type"})
TARGET = WORK / ("replication" if REPLICATION else "original")
TARGET.mkdir(parents=True, exist_ok=True)
VINTAGES = {
    "early_2018_publish": 10768,
    "middle_2021_publish": 1049,
    "late_2025_publish": 13192,
}
ZOOM = 17

def tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    latr = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(latr)) / math.pi) / 2.0 * n
    return x, y

def get_tile(layer, x, y):
    path = WORK / "tiles" / str(layer) / str(ZOOM) / str(x) / f"{y}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    def download():
        url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/{layer}/{ZOOM}/{y}/{x}"
        last = None
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=45, headers={"User-Agent":"ShanghaiUrbanVegetationValidation/1.0"})
                if r.ok and r.headers.get("content-type", "").startswith("image/"):
                    path.write_bytes(r.content); return
                last = RuntimeError(f"HTTP {r.status_code}, {r.headers.get('content-type')}")
            except requests.RequestException as exc:
                last = exc
            time.sleep(1.5 * (attempt + 1))
        raise last
    if not path.exists(): download()
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        path.unlink(missing_ok=True); download(); return Image.open(path).convert("RGB")

def patch_image(row, layer):
    fx, fy = tile_xy(row.centroid_lon, row.centroid_lat, ZOOM)
    tx, ty = int(fx), int(fy)
    mosaic = Image.new("RGB", (768, 768))
    for ix in range(tx-1, tx+2):
        for iy in range(ty-1, ty+2):
            mosaic.paste(get_tile(layer, ix, iy), ((ix-(tx-1))*256, (iy-(ty-1))*256))
    px = 256 + int((fx-tx)*256); py = 256 + int((fy-ty)*256)
    crop = mosaic.crop((px-192, py-192, px+192, py+192))
    d = ImageDraw.Draw(crop)
    d.ellipse((187,187,197,197), outline="yellow", width=2)
    d.line((192,177,192,207), fill="yellow", width=1); d.line((177,192,207,192), fill="yellow", width=1)
    return crop

font = ImageFont.load_default()
for start in range(0, len(PATCHES), 10):
    subset = PATCHES.iloc[start:start+10]
    sheet = Image.new("RGB", (3*384, len(subset)*410), "white")
    draw = ImageDraw.Draw(sheet)
    for rr, (_, row) in enumerate(subset.iterrows()):
        for cc, (label, layer) in enumerate(VINTAGES.items()):
            img = patch_image(row, layer)
            sheet.paste(img, (cc*384, rr*410+26))
            draw.text((cc*384+5, rr*410+5), f"{row.patch_id} | {label}", fill="black", font=font)
    sheet.save(TARGET / f"blind_contact_{start//10+1}.jpg", quality=92)

manifest = PATCHES[["patch_id","block_id","vegetation_type","centroid_lon","centroid_lat"]].copy()
manifest = manifest.rename(columns={"vegetation_type":"original_type"})
manifest["blinding"] = "No NDVI outcomes,  coefficients, or influence statistics displayed"
manifest["source_note"] = "Esri World Imagery Wayback versions; version release date may differ from imagery acquisition date"
manifest.to_csv(TARGET / "blind_validation_manifest.csv", index=False, encoding="utf-8-sig")
print({"contact_sheets": math.ceil(len(PATCHES)/10), "patches": len(PATCHES), "vintages": list(VINTAGES), "target":str(TARGET)})
