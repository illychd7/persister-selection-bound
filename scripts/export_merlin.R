# Export the MeRLin in vitro Seurat objects (GSE299711; CTRL and DT) to MatrixMarket + CSV.
# Requires R >= 4.3 with SeuratObject and Matrix.
# Usage: Rscript export_merlin.R <folder with GSM9044662_CTRL_filtered.rds and GSM9044663_DT_filtered.rds>
# Writes <folder>/{CTRL,DT}_counts.mtx (genes x cells), _genes.txt and _meta.csv.
suppressPackageStartupMessages({library(SeuratObject); library(Matrix)})
args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args) > 0) args[1] else "data/merlin"
files <- c(CTRL = "GSM9044662_CTRL_filtered.rds", DT = "GSM9044663_DT_filtered.rds")
for (s in names(files)) {
  o <- readRDS(file.path(dir, files[[s]]))
  counts <- LayerData(o, assay = "RNA", layer = "counts")
  writeMM(counts, file.path(dir, paste0(s, "_counts.mtx")))
  writeLines(rownames(counts), file.path(dir, paste0(s, "_genes.txt")))
  md <- o@meta.data; md$cell <- colnames(o)
  write.csv(md, file.path(dir, paste0(s, "_meta.csv")), row.names = FALSE)
  cat(s, nrow(counts), "genes x", ncol(counts), "cells\n")
}
