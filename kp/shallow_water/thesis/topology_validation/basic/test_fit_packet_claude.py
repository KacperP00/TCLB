import sys
import os
import glob
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
 
input_dir = sys.argv[1] if len(sys.argv) > 1 else "output/"
case_name = sys.argv[2] if len(sys.argv) > 2 else "unknown_case"
geom_name = sys.argv[3] if len(sys.argv) > 3 else "dense"
wave_lambda = float(sys.argv[4]) if len(sys.argv) > 4 else None
 
# Geometry presets. up/down = measurement strips, plane_in/plane_out = absorber entrance/exit.
# The downstream strip starts >= 250 LU behind the exit plane so that evanescent modes
# of the cavities (decay length ~ channel width / pi) are gone.
GEOMETRIES = {
    "dense": dict(y=(156, 306), up=(2250, 5250), down=(6900, 9750), plane_in=6000, plane_out=6600,
                  band=(300, 600), out="kp/shallow_water/thesis/wyniki_analizy_comparison_dense1.5",
                  obj=dict(refl=((2250, 2850), (63000, 97500)), trans=((6900, 7500), (51000, 97500)))),
    "old":   dict(y=(104, 204), up=(1500, 3500), down=(4650, 5500), plane_in=4000, plane_out=4400,
                  band=(200, 400), out="kp/shallow_water/thesis/wyniki_analizy_comparison", obj=None),
}
if geom_name not in GEOMETRIES:
    sys.exit(f"Unknown geometry '{geom_name}', use one of {list(GEOMETRIES)}")
G = GEOMETRIES[geom_name]
Y_CH_MIN, Y_CH_MAX = G["y"]
Y_MARGIN = 10
x_up_min, x_up_max = G["up"]
x_down_min, x_down_max = G["down"]
X_IN, X_OUT = G["plane_in"], G["plane_out"]
BAND_MIN, BAND_MAX = G["band"]
H_BASE = 9.55
C_WAVE = np.sqrt(0.00209424 * H_BASE)
VALID_FRAC = 0.10          # packet: trust bins with incident amplitude >= 10% of the band maximum
R2_MIN = 0.90              # fit quality below this is flagged
 
out_prefix = f"{G['out']}/{case_name}_RT"
os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
print(f"Geometry preset: {geom_name}; planes x_in={X_IN}, x_out={X_OUT}; mode: "
      f"{'single wave, lambda=%g' % wave_lambda if wave_lambda else 'wave packet'}")
 
 
def iter_of(path):
    return int(os.path.basename(path).split('_')[-1].split('.')[0])
 
 
def read_profile(path):
    """One VTK frame -> (x, rho averaged over channel width - h_base)."""
    mesh = pv.read(path)
    if hasattr(mesh, "dimensions"):
        if "Rho" in mesh.cell_data:
            arr, off = np.asarray(mesh.cell_data["Rho"]), 0.5
            nx, ny = mesh.dimensions[0] - 1, mesh.dimensions[1] - 1
        else:
            arr, off = np.asarray(mesh.point_data["Rho"]), 0.0
            nx, ny = mesh.dimensions[0], mesh.dimensions[1]
        if arr.size % (nx * ny) != 0:
            sys.exit(f"Unexpected Rho size {arr.size} for grid {nx}x{ny}")
        field = arr.reshape(-1, ny, nx)[0].astype(np.float64)
        x = mesh.origin[0] + (np.arange(nx) + off) * mesh.spacing[0]
        y = mesh.origin[1] + (np.arange(ny) + off) * mesh.spacing[1]
        rows = (y >= Y_CH_MIN + Y_MARGIN) & (y <= Y_CH_MAX - Y_MARGIN)
        return x, field[rows, :].mean(axis=0) - H_BASE
    if "Rho" in mesh.cell_data:
        pts, rho = mesh.cell_centers().points, np.asarray(mesh.cell_data["Rho"])
    else:
        pts, rho = mesh.points, np.asarray(mesh.point_data["Rho"])
    m = (pts[:, 1] >= Y_CH_MIN + Y_MARGIN) & (pts[:, 1] <= Y_CH_MAX - Y_MARGIN)
    x, inv = np.unique(pts[m, 0], return_inverse=True)
    return x, np.bincount(inv, weights=rho[m]) / np.bincount(inv) - H_BASE
 
 
# ---------------------------------------------------------------- read the frames
files = glob.glob(os.path.join(input_dir, "*.vti")) or glob.glob(os.path.join(input_dir, "*.pvti"))
files = sorted(files, key=iter_of)
if wave_lambda:
    period = wave_lambda / C_WAVE
    t_min = x_down_max / C_WAVE + 3 * period          # front has passed the downstream strip + 3 periods
    files = [f for f in files if iter_of(f) >= t_min]
    print(f"Single wave: period {period:.0f} it, frames from t >= {t_min:.0f}")
if len(files) < 10:
    sys.exit("Too few frames for the analysis")
print(f"Reading {len(files)} frames ...")
 
up_rows, dn_rows, its = [], [], []
for f in files:
    x_c, rho_c = read_profile(f)
    if x_c.min() > x_up_min or x_c.max() < x_down_max:
        sys.exit(f"Measurement strips ({x_up_min}-{x_down_max}) do not fit in the data "
                 f"(x = {x_c.min():.0f}..{x_c.max():.0f}). Wrong geometry preset '{geom_name}'?")
    mu = (x_c >= x_up_min) & (x_c <= x_up_max)
    md = (x_c >= x_down_min) & (x_c <= x_down_max)
    x_up, x_dn = x_c[mu], x_c[md]
    up_rows.append(rho_c[mu]); dn_rows.append(rho_c[md]); its.append(iter_of(f))
M_up, M_dn = np.array(up_rows), np.array(dn_rows)
t = np.array(its, float)
dts = np.diff(t)
if not np.allclose(dts, dts[0]):
    sys.exit(f"Non-uniform VTK frame spacing: {np.unique(dts)}")
dt = dts[0]
Nt = len(t)
print(f"Nt={Nt}, dt={dt:g}, Nx_up={len(x_up)}, Nx_down={len(x_dn)}, dx={np.diff(x_up)[0]:g}")
 
 
# ---------------------------------------------------------------- two-way complex-k fit
def fit_two_way(x, a, f):
    """a(x): complex amplitude at frequency f. Returns dict with kr, alpha, Af, Ab (at strip centre xc), r2."""
    xc = x.mean()
    xl = x - xc
    k0 = 2 * np.pi * f / C_WAVE
 
    def basis(kr, al):
        return np.column_stack([np.exp((-1j * kr - al) * xl), np.exp((1j * kr + al) * xl)])
 
    def solve(p):
        B = basis(*p)
        coef = np.linalg.lstsq(B, a, rcond=None)[0]
        return coef, B @ coef - a
 
    def resid(p):
        r = solve(p)[1]
        return np.concatenate([r.real, r.imag])
 
    # coarse start for the damping (avoids local minima), then refine kr and alpha
    best = min(((np.sum(resid([k0, al]) ** 2), al) for al in (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3)))
    res = least_squares(resid, [k0, best[1]], bounds=([0.85 * k0, 0.0], [1.15 * k0, 1e-2]),
                        x_scale=[0.01 * k0, 1e-4])
    kr, al = res.x
    (Af, Ab), r = solve(res.x)
    sst = np.sum(np.abs(a - a.mean()) ** 2)
    return dict(kr=kr, al=al, xc=xc, Af=Af, Ab=Ab, r2=1 - np.sum(np.abs(r) ** 2) / sst)
 
 
def amp_f(fit, xp):      # forward amplitude at plane xp
    return fit["Af"] * np.exp((-1j * fit["kr"] - fit["al"]) * (xp - fit["xc"]))
 
 
def amp_b(fit, xp):      # backward amplitude at plane xp
    return fit["Ab"] * np.exp((1j * fit["kr"] + fit["al"]) * (xp - fit["xc"]))
 
 
def evaluate(a_up, a_dn, f):
    fu, fd = fit_two_way(x_up, a_up, f), fit_two_way(x_dn, a_dn, f)
    inc = abs(amp_f(fu, X_IN))
    R = abs(amp_b(fu, X_IN)) / inc
    T = abs(amp_f(fd, X_OUT)) / inc
    Tbase = np.exp(-fu["al"] * (X_OUT - X_IN))
    return dict(f=f, lam=C_WAVE / f, R=R, T=T, D=1 - R**2 - T**2, Tbase=Tbase,
                D_excess=1 - R**2 - (T / Tbase) ** 2,
                R_down=abs(amp_b(fd, X_OUT)) / abs(amp_f(fd, X_OUT)),
                al_up=fu["al"], al_dn=fd["al"], c_up=2 * np.pi * f / fu["kr"], c_dn=2 * np.pi * f / fd["kr"],
                r2_up=fu["r2"], r2_dn=fd["r2"], inc=inc)
 
 
rows = []
if wave_lambda is None:
    # ------------------------------------------------------------- packet: every frequency bin in the band
    Xu = np.fft.rfft(M_up - M_up.mean(axis=0), axis=0)
    Xd = np.fft.rfft(M_dn - M_dn.mean(axis=0), axis=0)
    freqs = np.fft.rfftfreq(Nt, d=dt)
    for n in range(1, len(freqs)):
        lam = C_WAVE / freqs[n]
        if BAND_MIN <= lam <= BAND_MAX:
            rows.append(evaluate(Xu[n], Xd[n], freqs[n]))
    df = pd.DataFrame(rows).sort_values("lam").reset_index(drop=True)
    df["valid"] = (df["inc"] >= VALID_FRAC * df["inc"].max()) & (df["r2_up"] >= R2_MIN) & (df["r2_dn"] >= R2_MIN)
    # note: 'inc' is the incident amplitude at the entrance plane; threshold applies to the packet spectrum
else:
    # ------------------------------------------------------------- single wave: exact frequency
    f0 = 1.0 / period
    B = np.column_stack([np.cos(2 * np.pi * f0 * t), np.sin(2 * np.pi * f0 * t), np.ones_like(t)])
    pinv = np.linalg.pinv(B)
    cu, cd = pinv @ M_up, pinv @ M_dn
    a_up, a_dn = cu[0] - 1j * cu[1], cd[0] - 1j * cd[1]      # same sign convention as the FFT
    df = pd.DataFrame([evaluate(a_up, a_dn, f0)])
    df["valid"] = (df["r2_up"] >= R2_MIN) & (df["r2_dn"] >= R2_MIN)
 
v = df[df["valid"]]
if v.empty:
    sys.exit("No valid frequency bins (incident wave not found or fits poor) - see the CSV for the fit quality")
 
# ---------------------------------------------------------------- report
if wave_lambda is None:
    w = v["inc"] ** 2                                     # incident-energy weights
    mR = np.sqrt(np.sum(w * v["R"] ** 2) / w.sum()) * 100
    mT = np.sqrt(np.sum(w * v["T"] ** 2) / w.sum()) * 100
    head = f"Zakres wiarygodny: {v['lam'].min():.0f}-{v['lam'].max():.0f} LU ({len(v)} binow)\n"
else:
    mR, mT = v["R"].iloc[0] * 100, v["T"].iloc[0] * 100
    head = f"Fala ciagla, lambda = {wave_lambda:g} LU\n"
mD = 100 - (mR / 100) ** 2 * 100 - (mT / 100) ** 2 * 100
report = ("RAPORT R/T NA PLASZCZYZNACH ABSORBERA (dopasowanie dwukierunkowe)\n"
          "================================================================\n" + head +
          f"Plaszczyzny: wejscie x={X_IN}, wyjscie x={X_OUT} (odcinek {X_OUT - X_IN} LU)\n"
          f"Odbicie  R (amplitudowe): {mR:.2f}%\n"
          f"Transmisja T (amplitudowa): {mT:.2f}%\n"
          f"Dyssypacja D = 1-R^2-T^2: {mD:.2f}%  (zawiera tlumienie numeryczne odcinka: "
          f"T_baza = {100 * v['Tbase'].mean():.1f}%)\n"
          f"Tlumienie alpha (przed): {v['al_up'].mean():.2e} 1/LU, predkosc fazowa {v['c_up'].mean():.4f} "
          f"(c = {C_WAVE:.4f}); fala odbita za absorberem R_down = {100 * v['R_down'].mean():.1f}%\n"
          f"Jakosc dopasowania R^2: {v['r2_up'].min():.3f} (przed), {v['r2_dn'].min():.3f} (za)\n")
 
# Obj1/Obj2-like energies from the VTK data (height-deviation part, same zones and time gates as the xml)
if G["obj"] and wave_lambda is None:
    def gated_energy(M, xs, zone, gate):
        mx = (xs >= zone[0]) & (xs <= zone[1]); mt = (t >= gate[0]) & (t <= gate[1])
        return float(np.sum(M[np.ix_(mt, mx)] ** 2) * dt)
    (zr, gr), (zt, gt) = G["obj"]["refl"], G["obj"]["trans"]
    E_refl = gated_energy(M_up, x_up, zr, gr)
    E_trans = gated_energy(M_dn, x_dn, zt, gt)
    report += (f"Energie typu Obj (czesc wysokosciowa, jednostki wzgledne): "
               f"E_odbicie(Obj2) = {E_refl:.4e}, E_transmisja(Obj1) = {E_trans:.4e}\n")
print(report)
with open(f"{out_prefix}_report.txt", "w", encoding="utf-8") as fh:
    fh.write(report)
 
out = df.copy()
for col in ("R", "T", "D", "D_excess"):
    out.loc[~out["valid"], col] = np.nan
out.rename(columns={"lam": "Dlugosc_Fali"}).to_csv(f"{out_prefix}_widmo.csv", index=False)
 
if wave_lambda is None:
    plt.figure(figsize=(10, 6))
    plt.plot(v["lam"], 100 * v["R"], 'r--', lw=2, label='Odbicie R')
    plt.plot(v["lam"], 100 * v["T"], 'g-', lw=2, label='Transmisja T')
    plt.plot(v["lam"], 100 * v["D"], 'b:', lw=2, label='Dyssypacja D = 1-R²-T²')
    plt.plot(v["lam"], 100 * v["Tbase"], 'k--', lw=1, label='T_baza (tlumienie odcinka)')
    plt.xlabel("Dlugosc fali [LU]"); plt.ylabel("[%]"); plt.xlim(BAND_MIN, BAND_MAX); plt.ylim(0, 105)
    plt.title("R, T, D na plaszczyznach absorbera (bez normalizacji)")
    plt.legend(); plt.grid(True)
    plt.savefig(f"{out_prefix}_widmo.png", dpi=300); plt.close()
 