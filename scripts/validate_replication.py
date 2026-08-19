"""Outcome-blind historical validation of the fixed external-replication patches."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
TABLES = ROOT / "outputs" / "tables"
p=pd.read_csv(TABLES/"replication_patches.csv")
J={
"C03_07_tree_R1":("tree","PERSISTENT TREE","HIGH","wooded pond/park edge persists"),
"C03_07_grass_R1":("water","MIXED / UNRELIABLE","HIGH","aquaculture ponds, not grass"),
"C03_10_tree_R1":("tree","PERSISTENT TREE","HIGH","dense plantation persists"),
"C03_10_grass_R1":("crop","MIXED / UNRELIABLE","HIGH","greenhouse agriculture"),
"C06_14_tree_R1":("tree","PERSISTENT TREE","MEDIUM","tree strip/grove persists amid agriculture"),
"C06_14_grass_R1":("uncertain","MAJOR LAND-COVER CHANGE","HIGH","field cleared in middle period and pond/field reconfiguration late"),
"C07_07_tree_R1":("tree","PERSISTENT TREE","HIGH","mature residential canopy persists"),
"C07_07_grass_R1":("built","MAJOR LAND-COVER CHANGE","HIGH","construction/clearing followed by redesigned managed grounds"),
"C07_09_tree_R1":("uncertain","PERSISTENT VEGETATION, TYPE UNCERTAIN","MEDIUM","field/orchard/woody cover changes in structure"),
"C07_09_grass_R1":("crop","MIXED / UNRELIABLE","HIGH","greenhouse and row agriculture"),
"C09_04_tree_R1":("tree","PERSISTENT TREE","HIGH","riverbank/village grove persists"),
"C09_04_grass_R1":("crop","MIXED / UNRELIABLE","HIGH","nursery/row agriculture"),
"C09_07_tree_R1":("tree","PERSISTENT TREE","MEDIUM","interchange woodland/landscaping persists"),
"C09_07_grass_R1":("grass","PERSISTENT GRASS","HIGH","managed park grass persists"),
"C09_10_tree_R1":("tree","PERSISTENT TREE","HIGH","linear roadside woodland persists"),
"C09_10_grass_R1":("grass","PERSISTENT GRASS","MEDIUM","large open managed grass parcel persists"),
"C10_05_tree_R1":("tree","PERSISTENT TREE","HIGH","residential tree canopy persists"),
"C10_05_grass_R1":("grass","PERSISTENT GRASS","MEDIUM","managed riverfront/institutional grass persists"),
"C11_03_tree_R1":("built","MAJOR LAND-COVER CHANGE","HIGH","cleared site converted to urban park/infrastructure"),
"C11_03_grass_R1":("crop","MIXED / UNRELIABLE","HIGH","greenhouse agriculture"),
"C12_06_tree_R1":("tree","PERSISTENT TREE","HIGH","highway-edge wooded park persists"),
"C12_06_grass_R1":("built","MAJOR LAND-COVER CHANGE","HIGH","industrial/construction redevelopment"),
"C18_16_tree_R1":("crop","MIXED / UNRELIABLE","HIGH","row agriculture, not tree"),
"C18_16_grass_R1":("crop","MIXED / UNRELIABLE","HIGH","agricultural field"),
"C22_05_tree_R1":("uncertain","MIXED / UNRELIABLE","MEDIUM","developed agricultural/industrial mosaic"),
"C22_05_grass_R1":("crop","MIXED / UNRELIABLE","HIGH","greenhouse agriculture"),
"C22_07_tree_R1":("crop","MIXED / UNRELIABLE","HIGH","greenhouse agriculture"),
"C22_07_grass_R1":("crop","MIXED / UNRELIABLE","HIGH","agricultural fields"),
"C23_06_tree_R1":("crop","MIXED / UNRELIABLE","HIGH","agricultural field"),
"C23_06_grass_R1":("crop","MIXED / UNRELIABLE","HIGH","agricultural field"),
}
rows=[]
for r in p.itertuples(index=False):
    vt,status,conf,note=J[r.patch_id]
    include=status in ("PERSISTENT TREE","PERSISTENT GRASS") and conf in ("HIGH","MEDIUM")
    rows.append({**r._asdict(),"validated_type":vt,"historical_validation_status":status,
        "historical_confidence":conf,"major_change_flag":status=="MAJOR LAND-COVER CHANGE",
        "include_replication_confirmatory":include,
        "validation_source":"Esri World Imagery Wayback fixed 2018/2021/2025 released versions",
        "blind_validation":"No NDVI outcomes/effect estimates displayed","validation_notes":note})
o=pd.DataFrame(rows)
o.to_csv(OUT/"replication_patch_validation.csv",index=False,encoding="utf-8-sig")
print({"patches":len(o),"included":int(o.include_replication_confirmatory.sum()),
       "blocks":o.loc[o.include_replication_confirmatory,"block_id"].nunique(),
       "strata_blocks":pd.read_csv(TABLES/"replication_blocks.csv").merge(o[o.include_replication_confirmatory][["block_id"]].drop_duplicates()).urbanization_stratum.value_counts().to_dict(),
       "status":o.historical_validation_status.value_counts().to_dict()})
