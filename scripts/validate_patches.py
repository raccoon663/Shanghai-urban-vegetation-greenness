"""Materialize blinded historical-imagery judgments for all 50 study patches."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
TABLES = ROOT / "outputs" / "tables"
patches=pd.read_csv(OUT/"sampled_patches.csv")

# Judgments were made from the fixed 2018/2021/2025 Wayback triptychs without NDVI outcomes,
# then cross-checked against the outcome-blind annual LULC audit. Values: validated_type,
# final history status, early/mid/late, purity, confidence, management evidence, notes.
J={
"C01_07_tree_E1":("tree","PERSISTENT TREE","tree","tree","tree","MEDIUM","MEDIUM","none evident","wooded/orchard-like cover persists; late image denser"),
"C01_07_grass_E1":("grass","PERSISTENT GRASS","open grass","open grass","grass","MEDIUM","MEDIUM","possible managed field","open vegetated parcel persists; exact imagery acquisition dates uncertain"),
"C03_13_tree_E1":("tree","PERSISTENT TREE","tree","tree","tree","HIGH","HIGH","landscaped park","wooded park/islands visible in all versions"),
"C03_13_grass_E1":("crop","MIXED / UNRELIABLE","greenhouse crop","greenhouse crop","greenhouse crop","LOW","HIGH","irrigated/managed agriculture","greenhouse-dominated agricultural parcel, not grass"),
"C05_06_tree_E1":("tree","PERSISTENT TREE","tree plantation","tree plantation","tree plantation","HIGH","HIGH","managed plantation possible","dense woody plantation persists"),
"C05_06_grass_E1":("crop","MIXED / UNRELIABLE","greenhouse crop","greenhouse crop","greenhouse crop","LOW","HIGH","irrigated agriculture","greenhouses, not grass"),
"C05_09_tree_E1":("uncertain","PERSISTENT VEGETATION, TYPE UNCERTAIN","vegetated field","vegetated field","vegetated field","MEDIUM","MEDIUM","possible managed field","vegetation persists but tree structure is not consistently resolved"),
"C05_09_grass_E1":("grass","PERSISTENT GRASS","roadside grass","roadside grass","roadside grass","MEDIUM","MEDIUM","mowing likely","persistent transport-corridor verge"),
"C06_10_tree_E1":("uncertain","PERSISTENT VEGETATION, TYPE UNCERTAIN","vegetated field","vegetated field","vegetated field","MEDIUM","MEDIUM","managed parcel possible","persistent vegetation but tree type uncertain"),
"C06_10_grass_E1":("crop","MIXED / UNRELIABLE","greenhouse crop","greenhouse crop","greenhouse crop","LOW","HIGH","irrigated agriculture","greenhouse agriculture"),
"C07_04_tree_E1":("built","MAJOR LAND-COVER CHANGE","shore vegetation","cleared/construction","redevelopment","LOW","HIGH","none","substantial clearing and redevelopment"),
"C07_04_grass_E1":("grass","PERSISTENT GRASS","managed lawn","managed lawn","managed lawn","HIGH","HIGH","mowing/irrigation evident","large landscaped residential lawn"),
"C07_13_tree_E1":("tree","PERSISTENT TREE","park trees","park trees","park trees","HIGH","HIGH","landscaped park","sparse but persistent wooded park"),
"C07_13_grass_E1":("crop","MIXED / UNRELIABLE","greenhouse crop","greenhouse crop","greenhouse crop","LOW","HIGH","irrigated agriculture","greenhouse agriculture"),
"C07_16_tree_E1":("uncertain","MIXED / UNRELIABLE","vegetated parcel","cleared/bare","vegetated parcel","LOW","HIGH","none","clear middle-period disturbance; cover not historically reliable"),
"C07_16_grass_E1":("crop","MIXED / UNRELIABLE","orchard/crop","orchard/crop","orchard/crop","LOW","HIGH","managed agriculture","row crop/orchard texture"),
"C08_02_tree_E1":("tree","PERSISTENT TREE","tree grove","tree grove","tree grove","HIGH","HIGH","none evident","village-edge grove persists"),
"C08_02_grass_E1":("crop","MIXED / UNRELIABLE","greenhouse crop","greenhouse crop","greenhouse crop","LOW","HIGH","irrigated agriculture","greenhouse agriculture"),
"C09_12_tree_E1":("built","MAJOR LAND-COVER CHANGE","cleared site","construction","construction/bare","LOW","HIGH","none","construction and bare ground dominate"),
"C09_12_grass_E1":("crop","MIXED / UNRELIABLE","nursery/crop","nursery/crop","nursery/crop","LOW","HIGH","irrigated agriculture","regular planted rows indicate nursery/crop"),
"C09_15_tree_E1":("built","MAJOR LAND-COVER CHANGE","crop/field","crop/field","construction","LOW","HIGH","none","late-period conversion to construction"),
"C09_15_grass_E1":("uncertain","MIXED / UNRELIABLE","bare/paved","sparse vegetation","sparse vegetation","LOW","MEDIUM","none","low-purity disturbed parcel"),
"C10_02_tree_E1":("tree","PERSISTENT TREE","tree strip","tree strip","tree strip","HIGH","HIGH","managed shelterbelt possible","dense linear woodland persists"),
"C10_02_grass_E1":("crop","MIXED / UNRELIABLE","greenhouse crop","greenhouse crop","greenhouse crop","LOW","HIGH","irrigated agriculture","greenhouse agriculture"),
"C10_08_tree_E1":("tree","PERSISTENT TREE","urban park","urban park","urban park","HIGH","HIGH","landscaped/irrigated","mature urban park canopy persists"),
"C10_08_grass_E1":("uncertain","MAJOR LAND-COVER CHANGE","shore construction","redevelopment","redevelopment","LOW","HIGH","none","riverfront redevelopment"),
"C11_07_tree_E1":("tree","PERSISTENT TREE","urban trees","urban trees","urban trees","MEDIUM","MEDIUM","landscaped/irrigated","persistent institutional green space with mixed built edges"),
"C11_07_grass_E1":("grass","PERSISTENT GRASS","sports field","sports field","sports field","HIGH","HIGH","mowing/irrigation evident","persistent athletic field"),
"C11_09_tree_E1":("tree","PERSISTENT TREE","urban park","urban park","urban park","HIGH","HIGH","landscaped/irrigated","mature residential park canopy"),
"C11_09_grass_E1":("grass","PERSISTENT GRASS","sports field","sports field","sports field","HIGH","HIGH","mowing/irrigation evident","persistent stadium field"),
"C12_10_tree_E1":("tree","PERSISTENT TREE","urban forest","urban forest","urban forest","HIGH","HIGH","landscaped park","large wooded park persists"),
"C12_10_grass_E1":("uncertain","MIXED / UNRELIABLE","water/shore","construction/shore","shore vegetation","LOW","HIGH","none","water-edge construction mix"),
"C14_08_tree_E1":("uncertain","PERSISTENT VEGETATION, TYPE UNCERTAIN","open vegetation","open vegetation","shrub/open vegetation","MEDIUM","MEDIUM","none evident","persistent vegetation but tree structure uncertain"),
"C14_08_grass_E1":("built","MAJOR LAND-COVER CHANGE","industrial/crop","industrial/crop","construction/built","LOW","HIGH","none","late conversion/redevelopment"),
"C15_06_tree_E1":("tree","PERSISTENT TREE","residential trees","residential trees","residential trees","HIGH","HIGH","landscaped/irrigated","stable landscaped residential canopy"),
"C15_06_grass_E1":("grass","PERSISTENT GRASS","golf/lawn","golf/lawn","golf/lawn","HIGH","HIGH","mowing/irrigation evident","persistent managed golf-course grass"),
"C15_12_tree_E1":("crop","MIXED / UNRELIABLE","crop field","crop field","crop field","LOW","HIGH","managed agriculture","agricultural field, not tree"),
"C15_12_grass_E1":("uncertain","PERSISTENT VEGETATION, TYPE UNCERTAIN","wet field/crop","wet field/crop","wet field/crop","LOW","LOW","managed agriculture possible","grass versus crop cannot be separated reliably"),
"C16_07_tree_E1":("built","MAJOR LAND-COVER CHANGE","bare/agriculture","bare/agriculture","construction","LOW","HIGH","none","late construction conversion"),
"C16_07_grass_E1":("crop","MIXED / UNRELIABLE","greenhouse crop","greenhouse crop","greenhouse/built","LOW","HIGH","irrigated agriculture","greenhouses and later built disturbance"),
"C17_17_tree_E1":("uncertain","PERSISTENT VEGETATION, TYPE UNCERTAIN","orchard/crop","orchard/crop","woody vegetation","MEDIUM","MEDIUM","managed agriculture","persistent woody/agricultural cover; exact tree status uncertain"),
"C17_17_grass_E1":("crop","MIXED / UNRELIABLE","row crop","row crop","row crop","LOW","HIGH","managed agriculture","agricultural rows"),
"C20_13_tree_E1":("tree","PERSISTENT TREE","roadside trees","roadside trees","roadside trees","MEDIUM","MEDIUM","managed shelterbelt","persistent woody strip beside road"),
"C20_13_grass_E1":("grass","PERSISTENT GRASS","open field","open field","open field","MEDIUM","MEDIUM","possible managed field","persistent open grass-like cover"),
"C21_12_tree_E1":("tree","PERSISTENT TREE","tree plantation","tree plantation","tree plantation","HIGH","HIGH","managed plantation","persistent dense tree block"),
"C21_12_grass_E1":("uncertain","PERSISTENT VEGETATION, TYPE UNCERTAIN","field vegetation","field vegetation","field vegetation","LOW","LOW","managed agriculture possible","grass versus crop unresolved"),
"C22_10_tree_E1":("tree","PERSISTENT TREE","mosaic trees","mosaic trees","mosaic trees","MEDIUM","MEDIUM","managed orchard possible","persistent woody mosaic"),
"C22_10_grass_E1":("uncertain","PERSISTENT VEGETATION, TYPE UNCERTAIN","field vegetation","field vegetation","field vegetation","LOW","LOW","managed agriculture possible","grass versus crop unresolved"),
"C24_04_tree_E1":("crop","MIXED / UNRELIABLE","crop strips","crop strips","crop strips","LOW","HIGH","managed agriculture","agricultural strips, not tree"),
"C24_04_grass_E1":("water","MIXED / UNRELIABLE","aquaculture ponds","aquaculture ponds","aquaculture ponds","LOW","HIGH","managed aquaculture","pond complex, not grass"),
}

annual=pd.read_csv(OUT/"patch_year_landcover_audit.csv")
annual_summary=annual.groupby("patch_id").agg(
    annual_persistent_years=("landcover_temporal_status",lambda x:(x=="persistent_vegetation").sum()),
    annual_uncertain_years=("landcover_temporal_status",lambda x:x.astype(str).str.startswith("uncertain").sum()),
    annual_probable_change_years=("landcover_temporal_status",lambda x:(x=="probable_landcover_change").sum()),
).reset_index()
rows=[]
for p in patches.itertuples(index=False):
    vt, status, early, mid, late, purity, conf, management, notes=J[p.patch_id]
    persistent=status.startswith("PERSISTENT") and vt in ("tree","grass")
    rows.append({"patch_id":p.patch_id,"block_id":p.block_id,"original_type":p.vegetation_type,
        "validated_type":vt,"validation_status":status,"early_period_status":early,
        "mid_period_status":mid,"late_period_status":late,"major_change_flag":status=="MAJOR LAND-COVER CHANGE",
        "vegetation_purity_class":purity,"confidence":conf,
        "validation_source":"Esri World Imagery Wayback fixed 2018/2021/2025 released versions + Annual LULC v2 2017-2023 audit",
        "management_evidence":management,"notes":notes,
        "include_confirmatory":bool(persistent and conf in ("HIGH","MEDIUM")),
        "blind_validation":"NDVI outcomes/effect estimates/influence diagnostics not displayed"})
out=pd.DataFrame(rows).merge(annual_summary,on="patch_id",how="left")
out.to_csv(TABLES/"patch_validation.csv",index=False,encoding="utf-8-sig")
print({"patches":len(out),"status":out.validation_status.value_counts().to_dict(),
       "confidence":out.confidence.value_counts().to_dict(),"included":int(out.include_confirmatory.sum()),
       "included_blocks":out.loc[out.include_confirmatory,"block_id"].nunique()})
