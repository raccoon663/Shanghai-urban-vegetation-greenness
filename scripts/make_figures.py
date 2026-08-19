from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; O=ROOT/"outputs"; F=ROOT/"assets"/"figures"; F.mkdir(parents=True,exist_ok=True)
TABLES = ROOT / "outputs" / "tables"
plt.rcParams.update({"font.size":9,"figure.dpi":140,"axes.spines.top":False,"axes.spines.right":False})
def save(n,name):
    plt.tight_layout(); plt.savefig(F/f"figure_{n:02d}_{name}.png",dpi=300,bbox_inches="tight"); plt.savefig(F/f"figure_{n:02d}_{name}.pdf",bbox_inches="tight"); plt.close()
v=pd.read_csv(TABLES/"patch_validation.csv"); vp=pd.read_csv(O/"sampled_patches.csv")
v=v.merge(vp[["patch_id","centroid_lon","centroid_lat"]],on="patch_id",how="left")
rv=pd.read_csv(O/"replication_patch_validation.csv")
plt.figure(figsize=(6.8,5.4)); colors={"HIGH":"#1b9e77","MEDIUM":"#377eb8","LOW":"#e6550d"}
for c,g in v.groupby("confidence"):plt.scatter(g.centroid_lon,g.centroid_lat,c=colors.get(c,"gray"),label=f"Original {c}",s=32,marker="o",edgecolor="white")
for c,g in rv.groupby("historical_confidence"):plt.scatter(g.centroid_lon,g.centroid_lat,c=colors.get(c,"gray"),label=f"Replication {c}",s=42,marker="^",edgecolor="black",linewidth=.3)
plt.xlabel("Longitude");plt.ylabel("Latitude");plt.title("Historically validated vegetation patches");plt.legend(fontsize=7,ncol=2);save(1,"validated_patch_map")
plt.figure(figsize=(6.4,4)); x=pd.concat([v.assign(sample="Original").rename(columns={"confidence":"conf"}),rv.assign(sample="Replication").rename(columns={"historical_confidence":"conf"})]); pd.crosstab(x["conf"],x["sample"]).reindex(["HIGH","MEDIUM","LOW"]).plot.bar(ax=plt.gca(),color=["#377eb8","#1b9e77"]);plt.ylabel("Patches");plt.title("Validation confidence distribution");save(2,"validation_confidence")
c=pd.read_csv(TABLES/"platform_calibration.csv");plt.figure(figsize=(7,4)); lab=c.s2_platform.fillna("unknown")+" × "+c.landsat_platform;plt.errorbar(range(len(c)),c.slope,yerr=c.rmse,fmt="o",color="#377eb8",ecolor="#9ecae1");plt.axhline(1,color="k",ls="--");plt.xticks(range(len(c)),lab,rotation=55,ha="right");plt.ylabel("Slope (error bar = RMSE)");plt.title("Platform-specific cross-sensor calibration");save(3,"platform_calibration")
a=pd.read_csv(TABLES/"calibration_window_sensitivity.csv");plt.figure(figsize=(7,4));
for k,g in a.groupby(["s2_platform","landsat_platform"]):plt.plot(g.window_days,g.slope,marker="o",label=" × ".join(map(str,k)))
plt.axhline(1,color="k",ls="--");plt.xlabel("Matching window (days)");plt.ylabel("Slope");plt.title("Calibration temporal-window stability");plt.legend(fontsize=6,ncol=2);save(4,"annual_calibration_stability")
s=pd.read_csv(TABLES/"sensor_strategy_results.csv");q=s[s.endpoint=="summer_NDVI"]
plt.figure(figsize=(6.5,4));plt.errorbar(range(3),q.estimate,yerr=[q.estimate-q.ci_low,q.ci_high-q.estimate],fmt="o",color="#377eb8");plt.axhline(0,color="k",lw=.8);plt.xticks(range(3),["S2 only"," pooled","Platform-aware"]);plt.ylabel("Built-fraction coefficient");plt.title("Summer NDVI across predeclared sensor strategies");save(5,"sensor_strategies")
o=pd.read_csv(TABLES/"validated_primary_sample.csv");o=o[(o.sensor_strategy=="C_platform_aware")&o.include_confirmatory.astype(str).str.lower().eq("true")]
plt.figure(figsize=(5.8,4));plt.scatter(o.built_fraction_1000m,o.summer_NDVI,c="#377eb8",alpha=.6,s=18);z=np.polyfit(o.built_fraction_1000m,o.summer_NDVI,1);xx=np.linspace(o.built_fraction_1000m.min(),o.built_fraction_1000m.max());plt.plot(xx,np.polyval(z,xx),c="#d95f02");plt.xlabel("Built fraction (1000 m)");plt.ylabel("Summer NDVI");plt.title("Original validated sample");save(6,"original_summer_scatter")
r=pd.read_csv(O/"replication_primary_sample.csv");r=r[r.include_confirmatory.astype(str).str.lower().eq("true")];plt.figure(figsize=(5.8,4));plt.scatter(r.built_fraction_1000m,r.summer_NDVI,c="#1b9e77",alpha=.65,s=20);z=np.polyfit(r.built_fraction_1000m,r.summer_NDVI,1);xx=np.linspace(r.built_fraction_1000m.min(),r.built_fraction_1000m.max());plt.plot(xx,np.polyval(z,xx),c="#d95f02");plt.xlabel("Built fraction (1000 m)");plt.ylabel("Summer NDVI");plt.title("Independent replication");save(7,"replication_summer_scatter")
x=pd.read_csv(TABLES/"original_vs_replication.csv");plt.figure(figsize=(6,4));
for i,row in x.iterrows():plt.errorbar(row.estimate,i,xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],fmt="o",color="#377eb8" if row["sample"]=="original_validated" else "#1b9e77")
plt.axvline(0,color="k",lw=.8);plt.yticks(range(len(x)),x.endpoint+" / "+x["sample"]);plt.xlabel("Coefficient (95% cluster CI)");plt.title("Original versus replication");save(8,"original_vs_replication")
m=pd.read_csv(TABLES/"combined_primary_models.csv")
for n,ep in [(9,"summer_NDVI"),(10,"peak_NDVI")]:
 g=m[m.endpoint==ep];plt.figure(figsize=(6,4));plt.errorbar(g.estimate,range(len(g)),xerr=[g.estimate-g.ci_low,g.ci_high-g.estimate],fmt="o",color="#377eb8");plt.axvline(0,color="k",lw=.8);plt.yticks(range(len(g)),g.model);plt.xlabel("Coefficient (95% cluster CI)");plt.title(f"Combined final {ep.replace('_',' ')}");save(n,f"combined_{ep.lower()}")
pm=pd.read_csv(TABLES/"combined_primary_models.csv");full=pm[(pm.endpoint=="summer_NDVI")&(pm.model=="crop_adjusted")].iloc[0]
l=pd.read_csv(TABLES/"combined_leave_one_block_out.csv");plt.figure(figsize=(7,4.4))
ax=plt.gca()
for ep,g in l.groupby("endpoint"):ax.plot(range(len(g)),g.estimate,marker="o",ms=3,label=ep)
ax.axhline(0,color="k",lw=.8);ax.axvline(full.estimate,color="#9ecae1",ls=":",lw=1)
ax.text(0.5,0.97,f"Full-sample estimate ({full.estimate:+.3f})",transform=ax.transAxes,ha="center",va="top",fontsize=8,color="#1b7837",bbox=dict(facecolor="white",edgecolor="none",alpha=0.85,pad=1.8))
ax.set_xlabel("Left-out block iteration");ax.set_ylabel("Coefficient");ax.set_title("Final leave-one-block-out robustness");ax.legend(loc="lower right");save(11,"combined_loo")
t=pd.read_csv(TABLES/"thermal_validation.csv");fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,5.6),sharex=False)
colors=["#d95f02","#7570b3"]
for ax,row,c in zip((ax1,ax2),t.itertuples(index=False),colors):
 lo,hi=row.ci_low,row.ci_high;est=row.estimate
 ax.errorbar([est],[0],xerr=[[est-lo],[hi-est]],fmt="o",color=c,markersize=6,capsize=4)
 ax.axvline(0,color="k",lw=.8,ls="--")
 ax.set_yticks([0]);ax.set_yticklabels([])
 ax.set_ylim(-0.5,0.5)
 ax.set_xlim(lo-(hi-lo)*0.15,hi+(hi-lo)*0.15)
 ax.set_title(f"{row.relation.replace('_',' ')}  (n={row.n_block_years} block-years, {row.n_blocks} blocks)",fontsize=9)
ax2.set_xlabel("Coefficient (95% CI; units differ between panels)")
ax1.set_title("Secondary thermal consistency — built fraction → summer LST  (n=89 block-years, 16 blocks)",fontsize=10)
fig.text(0.5,0.015,"Directionally consistent supporting evidence; not a mechanistic or mediation test.",ha="center",fontsize=7.5,style="italic",color="gray")
fig.subplots_adjust(hspace=0.7,top=0.93,bottom=0.11,left=0.10,right=0.96)
F.mkdir(parents=True,exist_ok=True)
for ext,kw in [("png",{"dpi":300}),("pdf",{})]:
 fig.savefig(F/f"figure_12_thermal_consistency.{ext}",bbox_inches="tight",**kw)
plt.close(fig)
print("created",len(list(F.glob("*.png"))),"PNG and",len(list(F.glob("*.pdf"))),"PDF")
