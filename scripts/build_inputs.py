#!/usr/bin/env python3
"""Download GSE150949 (Oren et al. 2021, Watermelon) from GEO and build the input files used by
pc9_preprocess_keep.py, gene_panel_control.py and pooled_decomp.py:

  <out>/pc9_counts.npz                 cells x genes UMI counts (CSR, int32), cells in metadata order
  <out>/pc9_genes.txt                  gene symbols, one per line (no trailing newline)
  <out>/pooled_counts.npz              same for the pooled Watermelon experiment
  <out>/pooled_genes.txt
  <out>/GSE150949_metaData_with_lineage.txt.gz            (as downloaded)
  <out>/GSE150949_pooled_watermelon.metadata.matrix.csv.gz (as downloaded)

The GEO count matrices are dense CSV (genes x cells); they are parsed one gene at a time, so
memory stays bounded. Row i of each .npz equals row i of the corresponding metadata file
(checked below).

Usage: python build_inputs.py [--out data/pc9]
"""
from config import PC9  # data/results/figure folders, see config.py
import argparse, gzip, pathlib, shutil, urllib.request
import numpy as np, pandas as pd, scipy.sparse as sp

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE150nnn/GSE150949/suppl/"
FILES = {
    "pc9_counts": "GSE150949_pc9_count_matrix.csv.gz",
    "pc9_meta": "GSE150949_metaData_with_lineage.txt.gz",
    "pooled_counts": "GSE150949_pooled_watermelon.count.matrix.csv.gz",
    "pooled_meta": "GSE150949_pooled_watermelon.metadata.matrix.csv.gz",
}


def fetch(name, out):
    dst = out / name
    if not dst.exists():
        print("downloading", name, flush=True)
        with urllib.request.urlopen(GEO + name) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
    return dst


def dense_csv_to_csr(path):
    """Genes x cells dense CSV (first column = gene) -> cells x genes CSR, plus names."""
    rows, cols, vals, genes = [], [], [], []
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split(",")
        cells = [c.strip('"') for c in header]
        if cells[0] == "":                      # R write.csv: leading empty corner cell
            cells = cells[1:]
        for g, line in enumerate(f):
            name, rest = line.rstrip("\n").split(",", 1)
            v = np.fromstring(rest, sep=",", dtype=np.int64)
            if v.size != len(cells):
                raise ValueError(f"row {g} ({name}) has {v.size} values for {len(cells)} cells")
            nz = np.flatnonzero(v)
            rows.append(nz.astype(np.int32)); cols.append(np.full(nz.size, g, np.int32)); vals.append(v[nz].astype(np.int32))
            genes.append(name.strip('"'))
            if g % 5000 == 0:
                print(f"  {path.name}: {g} genes", flush=True)
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(cells), len(genes)), dtype=np.int32)
    return X, np.array(genes), np.array(cells)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=PC9)
    out = pathlib.Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    paths = {k: fetch(v, out) for k, v in FILES.items()}

    X, genes, cells = dense_csv_to_csr(paths["pc9_counts"])
    m = pd.read_csv(paths["pc9_meta"], sep="\t")
    assert list(m.index) == list(cells), "PC9 metadata and matrix cell order differ"
    sp.save_npz(out / "pc9_counts.npz", X)
    (out / "pc9_genes.txt").write_text("\n".join(genes))
    print("pc9", X.shape, "nnz", X.nnz, flush=True)

    X, genes, cells = dense_csv_to_csr(paths["pooled_counts"])
    m = pd.read_csv(paths["pooled_meta"])
    assert list(m["cell"]) == list(cells), "pooled metadata and matrix cell order differ"
    sp.save_npz(out / "pooled_counts.npz", X)
    (out / "pooled_genes.txt").write_text("\n".join(genes))
    print("pooled", X.shape, "nnz", X.nnz, flush=True)


if __name__ == "__main__":
    main()
