#!/usr/bin/env python3
"""Export the cell metadata (obs) of the authors' final ClonMapper TNBC object (GSE291678).

The h5ad on GEO is 9.5 GB; only its obs table is needed (sample, clone labels, QC flags).
A copy of the exported table is shipped as inputs/tnbc_authors_obs.csv.gz, so this step is
optional. Usage: python tnbc_obs_from_h5ad.py <GSE291678_barcoded-tnbc-final.h5ad>
"""
from config import TNBC
import sys
import h5py, numpy as np, pandas as pd


def read_obs(path):
    """Read the obs dataframe of an h5ad without loading X."""
    with h5py.File(path, "r") as h:
        g = h["obs"]
        idx_key = g.attrs.get("_index", "_index")
        cols = {}
        for k in g.keys():
            obj = g[k]
            if isinstance(obj, h5py.Group) and "categories" in obj:
                cats = [c.decode() if isinstance(c, bytes) else c for c in obj["categories"][()]]
                cols[k] = pd.Categorical.from_codes(obj["codes"][()], cats)
            elif isinstance(obj, h5py.Dataset):
                v = obj[()]
                if v.dtype.kind in "SO":
                    v = np.array([x.decode() if isinstance(x, bytes) else x for x in v])
                cols[k] = v
        index = cols.pop(idx_key)
        return pd.DataFrame(cols, index=index)


if __name__ == "__main__":
    obs = read_obs(sys.argv[1])
    obs.to_csv(TNBC + "/authors_obs.csv.gz")
    print("authors obs:", obs.shape)
