import sys
import os
import glob
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Argumenty: 1: katalog VTI, 2: nazwa przypadku, 3: mnoznik skali (np. 1.0, 1.5, 2.0),
# 4: katalog wyjsciowy wynikow, 5 (opcjonalny): bazowa dlugosc fali ciaglej na siatce 1.0x (np. 324.0)
input_dir = sys.argv[1] if len(sys.argv) > 1 else "output/"
case_name = sys.argv[2] if len(sys.argv) > 2 else "unknown_case"
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
out_dir = sys.argv[4] if len(sys.argv) > 4 else f"kp/shallow_water/thesis/wyniki_fit_{scale}x"
base_wave_lambda = float(sys.argv[5]) if len(sys.argv) > 5 else None

wave_lambda = base_wave_lambda * scale if base_wave_lambda is not None else None

# Automatyczne skalowanie stref pomiarowych i plaszczyzn odniesienia z siatki bazowej 1.0x
Y_CH_MIN, Y_CH_MAX = int(round(104 * scale)), int(round(204 * scale))
Y_MARGIN = max(2, int(round(10 * scale)))
x_up_min, x_up_max = int(round(1500 * scale)), int(round(3500 * scale))
x_down_min, x_down_max = int(round(4650 * scale)), int(round(6200 * scale))
X_IN, X_OUT = 4000.0 * scale, 4400.0 * scale
BAND_MIN, BAND_MAX = 200.0 * scale, 400.0 * scale

H_BASE = 9.55
C_WAVE = np.sqrt(0.00209424 * H_BASE)
VALID_FRAC = 0.10
R2_MIN = 0.90

os.makedirs(out_dir, exist_ok=True)
out_prefix = f"{out_dir}/{case_name}_RT"
print(f"Skala: {scale}x; plaszczyzny x_in={X_IN:.0f}, x_out={X_OUT:.0f}; tryb: "
      f"{'fala ciagla, lambda=%g' % wave_lambda if wave_lambda else 'paczka falowa'}")

def iter_of(path):
    return int(os.path.basename(path).split('_')[-1].split('.')[0])

def read_profile(path):
    # Ekstrakcja profilu sredniego po szerokosci kanalu z pojedynczej klatki VTK
    mesh = pv.read(path)
    if hasattr(mesh, "dimensions"):
        if "Rho" in mesh.cell_data:
            arr, off = np.asarray(mesh.cell_data["Rho"]), 0.5
            nx, ny = mesh.dimensions[0] - 1, mesh.dimensions[1] - 1
        else:
            arr, off = np.asarray(mesh.point_data["Rho"]), 0.0
            nx, ny = mesh.dimensions[0], mesh.dimensions[1]
        field = arr.reshape(-1, ny, nx)[0].astype(np.float64)
        x = mesh.origin[0] + (np.arange(nx) + off) * mesh.spacing[0]
        y = mesh.origin[1] + (np.arange(ny) + off) * mesh.spacing[1]
        rows = (y >= Y_CH_MIN + Y_MARGIN) & (y <= Y_CH_MAX - Y_MARGIN)
        return x, field[rows, :].mean(axis=0) - H_BASE
    pts = mesh.cell_centers().points if "Rho" in mesh.cell_data else mesh.points
    rho = np.asarray(mesh.cell_data["Rho"] if "Rho" in mesh.cell_data else mesh.point_data["Rho"])
    m = (pts[:, 1] >= Y_CH_MIN + Y_MARGIN) & (pts[:, 1] <= Y_CH_MAX - Y_MARGIN)
    x, inv = np.unique(pts[m, 0], return_inverse=True)
    return x, np.bincount(inv, weights=rho[m]) / np.bincount(inv) - H_BASE

files = glob.glob(os.path.join(input_dir, "*.vti")) or glob.glob(os.path.join(input_dir, "*.pvti"))
files = sorted(files, key=iter_of)
if wave_lambda:
    period = wave_lambda / C_WAVE
    t_min = x_down_max / C_WAVE + 3 * period
    files = [f for f in files if iter_of(f) >= t_min]
if len(files) < 10:
    sys.exit("Zbyt malo klatek VTK do analizy.")

up_rows, dn_rows, its = [], [], []
for f in files:
    x_c, rho_c = read_profile(f)
    mu = (x_c >= x_up_min) & (x_c <= x_up_max)
    md = (x_c >= x_down_min) & (x_c <= x_down_max)
    x_up, x_dn = x_c[mu], x_c[md]
    up_rows.append(rho_c[mu]); dn_rows.append(rho_c[md]); its.append(iter_of(f))

M_up, M_dn = np.array(up_rows), np.array(dn_rows)
t = np.array(its, float)
dt = np.diff(t)[0]
Nt = len(t)

def fit_two_way(x, a, f, fixed_kr_al=None):
    # Dwukierunkowe dopasowanie falowe z rzutowaniem zmiennych (Golub-Pereyra)
    xc = x.mean()
    xl = x - xc
    k0 = 2 * np.pi * f / C_WAVE

    def basis(kr, al):
        return np.column_stack([np.exp((-1j * kr - al) * xl), np.exp((1j * kr + al) * xl)])

    def solve(p):
        B = basis(*p)
        coef = np.linalg.lstsq(B, a, rcond=None)[0]
        return coef, B @ coef - a

    if fixed_kr_al is None:
        def resid(p):
            r = solve(p)[1]
            return np.concatenate([r.real, r.imag])

        best = min(((np.sum(resid([k0, al]) ** 2), al) for al in (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3)))
        res = least_squares(resid, [k0, best[1]], bounds=([0.85 * k0, 0.0], [1.15 * k0, 1e-2]),
                            x_scale=[0.01 * k0, 1e-4])
        kr, al = res.x
    else:
        kr, al = fixed_kr_al

    (Af, Ab), r = solve([kr, al])
    sst = np.sum(np.abs(a - a.mean()) ** 2) + 1e-30
    return dict(kr=kr, al=al, xc=xc, Af=Af, Ab=Ab, r2=max(0.0, 1.0 - np.sum(np.abs(r) ** 2) / sst))

def amp_f(fit, xp):
    return fit["Af"] * np.exp((-1j * fit["kr"] - fit["al"]) * (xp - fit["xc"]))

def amp_b(fit, xp):
    return fit["Ab"] * np.exp((1j * fit["kr"] + fit["al"]) * (xp - fit["xc"]))

def evaluate(a_up, a_dn, f):
    # Wyznaczenie liczby falowej i tlumienia przed przeszkoda oraz dopasowanie amplitud za przeszkoda
    fu = fit_two_way(x_up, a_up, f)
    fd = fit_two_way(x_dn, a_dn, f, fixed_kr_al=(fu["kr"], fu["al"]))
    inc = abs(amp_f(fu, X_IN))
    R = abs(amp_b(fu, X_IN)) / inc
    T = abs(amp_f(fd, X_OUT)) / inc
    Tbase = np.exp(-fu["al"] * (X_OUT - X_IN))
    T_comp = T / Tbase
    return dict(
        f=f, lam=C_WAVE / f, lam_base_1x=(C_WAVE / f) / scale,
        R=R, T=T, T_comp=T_comp, D=1 - R**2 - T**2, Tbase=Tbase,
        D_excess=1 - R**2 - T_comp**2,
        R_down=abs(amp_b(fd, X_OUT)) / (abs(amp_f(fd, X_OUT)) + 1e-30),
        al_up=fu["al"], c_up=2 * np.pi * f / fu["kr"],
        r2_up=fu["r2"], r2_dn=fd["r2"], inc=inc
    )

rows = []
if wave_lambda is None:
    Xu = np.fft.rfft(M_up - M_up.mean(axis=0), axis=0)
    Xd = np.fft.rfft(M_dn - M_dn.mean(axis=0), axis=0)
    freqs = np.fft.rfftfreq(Nt, d=dt)
    for n in range(1, len(freqs)):
        lam = C_WAVE / freqs[n]
        if BAND_MIN <= lam <= BAND_MAX:
            rows.append(evaluate(Xu[n], Xd[n], freqs[n]))
    df = pd.DataFrame(rows).sort_values("lam").reset_index(drop=True)
    # Kwalifikacja binu: wysoka amplituda padajaca i dobre dopasowanie w strefie przed falochronem
    df["valid"] = (df["inc"] >= VALID_FRAC * df["inc"].max()) & (df["r2_up"] >= R2_MIN)
else:
    f0 = 1.0 / period
    B = np.column_stack([np.cos(2 * np.pi * f0 * t), np.sin(2 * np.pi * f0 * t), np.ones_like(t)])
    pinv = np.linalg.pinv(B)
    cu, cd = pinv @ M_up, pinv @ M_dn
    a_up, a_dn = cu[0] - 1j * cu[1], cd[0] - 1j * cd[1]
    df = pd.DataFrame([evaluate(a_up, a_dn, f0)])
    df["valid"] = df["r2_up"] >= R2_MIN

v = df[df["valid"]]
if v.empty:
    sys.exit("Brak wiarygodnych binow czestotliwosciowych.")

if wave_lambda is None:
    w = v["inc"] ** 2
    mR = np.sqrt(np.sum(w * v["R"] ** 2) / w.sum()) * 100
    mT = np.sqrt(np.sum(w * v["T"] ** 2) / w.sum()) * 100
    mT_comp = np.sqrt(np.sum(w * v["T_comp"] ** 2) / w.sum()) * 100
    head = f"Zakres wiarygodny: {v['lam'].min():.0f}-{v['lam'].max():.0f} LU ({len(v)} binow, skala {scale}x)\n"
else:
    mR = v["R"].iloc[0] * 100
    mT = v["T"].iloc[0] * 100
    mT_comp = v["T_comp"].iloc[0] * 100
    head = f"Fala ciagla, lambda = {wave_lambda:g} LU (skala {scale}x)\n"

mD = 100 - (mR / 100) ** 2 * 100 - (mT / 100) ** 2 * 100
mD_excess = 100 - (mR / 100) ** 2 * 100 - (mT_comp / 100) ** 2 * 100

report = (
    "RAPORT R/T NA PLASZCZYZNACH ABSORBERA (dopasowanie dwukierunkowe)\n"
    "================================================================\n" + head +
    f"Plaszczyzny: wejscie x={X_IN:.0f}, wyjscie x={X_OUT:.0f} (odcinek {X_OUT - X_IN:.0f} LU)\n"
    f"Odbicie R (amplitudowe):                 {mR:.2f}%\n"
    f"Transmisja T surowa (krawedzie):         {mT:.2f}% (T_baza odcinka = {100 * v['Tbase'].mean():.2f}%)\n"
    f"Transmisja T_comp (skompensowana):       {mT_comp:.2f}%\n"
    f"Dyssypacja calkowita D = 1-R^2-T^2:      {mD:.2f}%\n"
    f"Absorpcja netto falochronu (D_excess):   {mD_excess:.2f}%\n"
    f"Tlumienie alpha (przed): {v['al_up'].mean():.2e} 1/LU, predkosc fazowa {v['c_up'].mean():.4f} (c = {C_WAVE:.4f})\n"
    f"Jakosc dopasowania R^2 (przed): min {v['r2_up'].min():.4f}\n"
)

print(report)
with open(f"{out_prefix}_report.txt", "w", encoding="utf-8") as fh:
    fh.write(report)

out = df.copy()
for col in ("R", "T", "T_comp", "D", "D_excess"):
    out.loc[~out["valid"], col] = np.nan
out.rename(columns={"lam": "Dlugosc_Fali", "lam_base_1x": "Dlugosc_Fali_1x"}).to_csv(f"{out_prefix}_widmo.csv", index=False)