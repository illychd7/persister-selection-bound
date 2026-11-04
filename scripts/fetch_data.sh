#!/usr/bin/env bash
# Download every public input from GEO into data/ (about 3.5 GB) and export the MeRLin objects.
# Usage: bash scripts/fetch_data.sh        (set PSB_DATA to use another data folder)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${PSB_DATA:-$HERE/../data}"
mkdir -p "$DATA/geo/raw" "$DATA/merlin" "$DATA/tnbc"
get() { [ -s "$2" ] || curl -fL --retry 3 -o "$2" "$1"; }
S=https://ftp.ncbi.nlm.nih.gov/geo/samples

# 1. PC9 / osimertinib and pooled Watermelon (GSE150949); writes data/pc9
python "$HERE/build_inputs.py"

# 2. ClonMapper MDA-MB-231 under doxorubicin: per-sample Cell Ranger matrices (GSE291678)
i=8839788
for code in PT-1 PT-2 Dox A1 A4 A6 B2 B5 B6; do
  for part in barcodes.tsv.gz features.tsv.gz matrix.mtx.gz; do
    get "$S/GSM8839nnn/GSM$i/suppl/GSM${i}_${code}_${part}" "$DATA/geo/raw/GSM${i}_${code}_${part}"
  done
  i=$((i + 1))
done
# authors' cell metadata: shipped in inputs/; to rebuild it, download the 9.5 GB h5ad and run
#   python scripts/tnbc_obs_from_h5ad.py GSE291678_barcoded-tnbc-final.h5ad
[ -s "$DATA/tnbc/authors_obs.csv.gz" ] || cp "$HERE/../inputs/tnbc_authors_obs.csv.gz" "$DATA/tnbc/authors_obs.csv.gz"

# 3. ClonMapper targeted barcode sequencing (GSE291679)
get https://ftp.ncbi.nlm.nih.gov/geo/series/GSE291nnn/GSE291679/suppl/GSE291679_counts-treated.tsv.gz \
    "$DATA/geo/GSE291679_counts-treated.tsv.gz"

# 4. MeRLin WM4237-1 in vitro, control and dabrafenib + trametinib (GSE299711)
get "$S/GSM9044nnn/GSM9044662/suppl/GSM9044662_CTRL_filtered.rds" "$DATA/merlin/GSM9044662_CTRL_filtered.rds"
get "$S/GSM9044nnn/GSM9044663/suppl/GSM9044663_DT_filtered.rds" "$DATA/merlin/GSM9044663_DT_filtered.rds"
Rscript "$HERE/export_merlin.R" "$DATA/merlin"
echo "done: $DATA"
