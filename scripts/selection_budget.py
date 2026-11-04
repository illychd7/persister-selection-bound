"""Observed clone-level enrichment from targeted barcode-seq (GSE291679).
Enrichment of clone c in well w = percent_w(c) / percent_PT(c).
Clone-level enrichment is a LOWER bound on state-level selection sup g / E g."""
from config import GEO, RESULTS  # data/results/figure folders, see config.py
import pandas as pd, numpy as np
df = pd.read_csv(GEO + "/GSE291679_counts-treated.tsv.gz", sep="\t")
df = df[df.barcode != "unknown"]
pt = df[df["sample"] == "JA21458-1KB3"].set_index("barcode")["percent"]
rows = []
for w, g in df[df["sample"] != "JA21458-1KB3"].groupby("sample"):
    g = g[g.barcode.isin(pt.index)].copy()
    g["enrich"] = g["percent"] / pt.loc[g.barcode].values
    well = w.split("-")[1]
    dose = {"A": 250, "B": 400, "C": 550}[well[0]]
    top = g.sort_values("percent", ascending=False)
    rows.append(dict(well=well, dose_nM=dose, clones_detected=len(g),
                     clones_surviving_pct=100 * len(g) / len(pt),
                     top3_share_pct=top["percent"].head(3).sum(),
                     max_clone_enrichment=g["enrich"].max(),
                     enrich_of_dominant=top["enrich"].iloc[0]))
out = pd.DataFrame(rows).sort_values(["dose_nM", "well"])
print("baseline clones:", len(pt), " baseline max share %:", pt.max().round(2))
print(out.round(2).to_string(index=False))
out.to_csv(RESULTS + "/selection_budget_barcodeseq.csv", index=False)
