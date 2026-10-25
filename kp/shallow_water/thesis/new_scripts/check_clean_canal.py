# -*- coding: utf-8 -*-
# Model-independent check of a CLEAN CANAL packet run: wave amplitude and arrival time vs x.
# Usage: python3 check_clean_canal.py <vti_dir> [dense|old]
# Ideal inviscid canal: amplitude stays ~1.0 along x and c_eff = sqrt(g*h) = 0.1414.
import sys, os, glob
import numpy as np
import pyvista as pv

vti_dir = sys.argv[1] if len(sys.argv) > 1 else "output/"
geom = sys.argv[2] if len(sys.argv) > 2 else "dense"
Y = {"dense": (156, 306), "old": (104, 204)}[geom]
XS = list(range(500, 8001 if geom == "old" else 12001, 500))
H_BASE, MARGIN = 9.55, 10

files = glob.glob(os.path.join(vti_dir, "*.vti"))
it = lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0])
files = sorted(files, key=it)
t = np.array([it(f) for f in files], float)

rows = []
for f in files:
    m = pv.read(f)
    if "Rho" in m.cell_data:
        a, nx, ny, off = np.asarray(m.cell_data["Rho"]), m.dimensions[0]-1, m.dimensions[1]-1, 0.5
    else:
        a, nx, ny, off = np.asarray(m.point_data["Rho"]), m.dimensions[0], m.dimensions[1], 0.0
    fld = a.reshape(-1, ny, nx)[0].astype(float)
    y = m.origin[1] + (np.arange(ny) + off) * m.spacing[1]
    sel = (y >= Y[0] + MARGIN) & (y <= Y[1] - MARGIN)
    rows.append(fld[sel].mean(0) - H_BASE)
Mx = np.array(rows)                                   # [t, x]
c = np.sqrt(0.00209424 * H_BASE)
print(f"frames={len(t)}  dt={np.diff(t)[0]:g}  nx={Mx.shape[1]}   c = {c:.4f}")
print("     x   amplitude (rel. x=500)   centroid time   c_eff")
a0 = prev = None
for x in XS:
    if x + 5 > Mx.shape[1]: break
    s = Mx[:, x-5:x+5].mean(1); e = (s**2).sum(); a = np.sqrt(e); tc = (t*s**2).sum()/e
    a0 = a if a0 is None else a0
    ce = "" if prev is None else f"{500/(tc - prev):.4f}"
    print(f"{x:6d}   {a/a0:10.3f}              {tc:9.0f}     {ce}")
    prev = tc
