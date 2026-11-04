"""Digitize Oren et al. (bioRxiv 2020) Fig. 1b 'Divisions all' heatmap.
Each row = one tracked PC9 lineage (1,135 rows), x = days 0-14 under 300 nM osimertinib,
colour = cumulative number of divisions (colorbar 0..7), white = lineage perished."""
import sys, numpy as np
from PIL import Image
heat = np.asarray(Image.open(sys.argv[1]).convert('RGB')).astype(float)
cbar = np.asarray(Image.open(sys.argv[2]).convert('RGB')).astype(float)
H, W, _ = heat.shape
cb = cbar[::8, cbar.shape[1]//2, :]              # top=7, bottom=0 (subsampled)
cbval = np.linspace(7, 0, cb.shape[0])
white = heat.min(axis=2) > 235
# map coloured pixels to division counts via nearest colorbar colour
flat = heat.reshape(-1, 3)
idx = np.empty(flat.shape[0], int)
for s in range(0, flat.shape[0], 50000):
    d = ((flat[s:s+50000, None, :] - cb[None, :, :])**2).sum(-1)
    idx[s:s+50000] = d.argmin(1)
div = np.rint(cbval[idx]).reshape(H, W)
div[white] = np.nan
N = 1135
edges = np.linspace(0, H, N + 1)
days = np.array([3, 7, 14])
cols = np.clip((days / 14 * W).astype(int) - 1, 0, W - 1)
rows = []
for i in range(N):
    a, b = int(round(edges[i])), int(round(edges[i + 1]))
    blk = div[a:b]
    # a lineage row is alive at column c if the majority of its pixel rows are coloured
    alive = (~np.isnan(blk)).mean(0) > 0.5
    lastcol = np.where(alive)[0].max() if alive.any() else -1
    dv = np.nanmedian(blk, axis=0)
    rows.append((lastcol, [alive[c] for c in cols], [dv[c] if alive[c] else np.nan for c in cols]))
last = np.array([r[0] for r in rows]); A = np.array([r[1] for r in rows]); D = np.array([r[2] for r in rows], float)
print(f"rows={N} image={W}x{H}")
for j, t in enumerate(days):
    fa = A[:, j].mean(); dmax = np.nanmax(D[:, j]); dq = np.nanpercentile(D[:, j], 99)
    # mean progeny proxy: alive lineages contribute 2^divisions (tracked path)
    prog = np.where(A[:, j], 2.0 ** np.nan_to_num(D[:, j]), 0.0)
    print(f"day {t:2d}: alive={A[:,j].sum():4d} ({fa:.3f})  max_div={dmax:.0f}  p99_div={dq:.0f}  "
          f"mean_progeny_proxy={prog.mean():.3f}  budget_conservative=2^dmax/f_alive={2**dmax/fa:.1f}  "
          f"budget_proxy=max/mean={prog.max()/prog.mean():.1f}")
np.savez(sys.argv[3], last=last, A=A, D=D, days=days, W=W)
