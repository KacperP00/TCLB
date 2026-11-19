import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

BM_DIR = "projects/aerospace_injector/benchmarks"
ST_NOMINAL = 0.090
UC_REF = 0.028


def latest_vti(case_dir):
  # Pobranie ostatniego pliku VTI z danego katalogu benchmarku
  files = sorted(glob.glob(os.path.join(case_dir, "output", "*_VTK_P00_*.vti")))
  return files[-1] if files else None


def extract_zero_contour(pf_2d, y_min_idx=4, y_max_idx=None):
  # Podpikselowa ekstrakcja izokontury phi = 0.0 z odcieciem obu scian
  ny = pf_2d.shape[0]
  if y_max_idx is None:
    y_max_idx = ny - 3
  sub = pf_2d[y_min_idx:y_max_idx, :]

  p0 = sub[:, :-1]
  p1 = sub[:, 1:]
  mask_h = ((p0 * p1) <= 0.0) & (p0 != p1)
  jh, ih = np.where(mask_h)
  th = -p0[jh, ih] / (p1[jh, ih] - p0[jh, ih])
  xh, yh = ih + th, jh + y_min_idx

  q0 = sub[:-1, :]
  q1 = sub[1:, :]
  mask_v = ((q0 * q1) <= 0.0) & (q0 != q1)
  jv, iv = np.where(mask_v)
  tv = -q0[jv, iv] / (q1[jv, iv] - q0[jv, iv])
  xv, yv = iv.astype(float), jv + y_min_idx + tv

  return np.concatenate([xh, xv]), np.concatenate([yh, yv])


# --- 1. ANALIZA BENCHMARKU YOUNGA-LAPLACE'A ---
laplace_rows = []
print("=== BENCHMARK 1: PRAWO YOUNGA-LAPLACE'A (ST_nom = 0.090) ===")
print(
    f"{'R_init':>6s} | {'R_eq':>6s} | {'dP [LU]':>9s} | {'rho_in(std)':>17s} |"
    f" {'rho_out(std)':>17s} | {'max|U_sp|':>9s}"
)
print("-" * 80)

for r_init in [15, 20, 30, 40]:
  vti = latest_vti(os.path.join(BM_DIR, f"laplace_R{r_init}"))
  if not vti:
    continue
  m = pv.read(vti)
  ny, nx = m.dimensions[1] - 1, m.dimensions[0] - 1
  pf = m["PhaseField"].reshape((ny, nx))
  rho = m["Rho"].reshape((ny, nx))
  u = m["U"].reshape((ny, nx, 3))
  umag = np.linalg.norm(u, axis=2)

  area = float(np.sum(pf > 0.0))
  r_eq = np.sqrt(area / np.pi)

  yy, xx = np.mgrid[0:ny, 0:nx]
  r_grid = np.hypot(xx - (nx / 2.0 - 0.5), yy - (ny / 2.0 - 0.5))
  mask_in = r_grid < max(4.0, r_eq - 7.0)
  mask_out = r_grid > r_eq + 12.0

  rho_in = float(np.mean(rho[mask_in]))
  std_in = float(np.std(rho[mask_in]))
  rho_out = float(np.mean(rho[mask_out]))
  std_out = float(np.std(rho[mask_out]))
  dp = (rho_in - rho_out) / 3.0
  u_spur = float(np.max(umag))

  laplace_rows.append({
      "R_init": r_init,
      "R_eq": round(r_eq, 3),
      "inv_R": round(1.0 / r_eq, 6),
      "dP": round(dp, 6),
      "rho_in": round(rho_in, 6),
      "rho_out": round(rho_out, 6),
      "std_in": round(std_in, 6),
      "std_out": round(std_out, 6),
      "u_spur": round(u_spur, 6),
  })
  print(
      f"{r_init:6d} | {r_eq:6.2f} | {dp:9.6f} |"
      f" {rho_in:.5f}({std_in:.1e}) | {rho_out:.5f}({std_out:.1e}) |"
      f" {u_spur:9.6f}"
  )

inv_rs = np.array([r["inv_R"] for r in laplace_rows])
dps = np.array([r["dP"] for r in laplace_rows])
sigma_num, intercept = np.polyfit(inv_rs, dps, 1)
r2_lap = float(np.corrcoef(inv_rs, dps)[0, 1] ** 2)
sigma_err_pct = 100.0 * (sigma_num - ST_NOMINAL) / ST_NOMINAL
print(
    f"Regresja dP = {sigma_num:.5f}*(1/R) + ({intercept:+.6f}) | R^2={r2_lap:.6f}"
    f" | Blad wzgledem 0.090: {sigma_err_pct:+.2f}%"
)

# --- 2. ANALIZA BENCHMARKU KATA ZWILZANIA (SESSILE DROP) ---
sessile_rows = []
sessile_contours = []
print("\n=== BENCHMARK 2: KAT ZWILZANIA (WallAdhesionDecay = 0.7) ===")
print(
    f"{'Zadany [deg]':>12s} | {'Zmierzony [deg]':>15s} | {'Blad [deg]':>10s} |"
    f" {'R_fit [LU]':>10s} | {'RMS [LU]':>8s} | {'max|U_sp|':>10s}"
)
print("-" * 78)

y_wall = 0.5
for deg_label, rad_val in [(60, 1.0472), (90, 1.5708), (143, 2.5000)]:
  theta_set = np.rad2deg(rad_val)
  vti = latest_vti(os.path.join(BM_DIR, f"sessile_{deg_label}deg"))
  if not vti:
    continue
  m = pv.read(vti)
  ny, nx = m.dimensions[1] - 1, m.dimensions[0] - 1
  pf = m["PhaseField"].reshape((ny, nx))
  u = m["U"].reshape((ny, nx, 3))
  u_spur = float(np.max(np.linalg.norm(u[1:-1, :], axis=2)))

  # Ekstrakcja wylacznie konturu kropli z pominieciem dolnej i gornej sciany
  xc, yc = extract_zero_contour(pf, y_min_idx=4, y_max_idx=ny - 3)
  A = np.column_stack([2.0 * xc, 2.0 * yc, np.ones_like(xc)])
  sol, *_ = np.linalg.lstsq(A, xc**2 + yc**2, rcond=None)
  x0, y0 = float(sol[0]), float(sol[1])
  r_fit = float(np.sqrt(sol[2] + x0**2 + y0**2))
  rms_fit = float(np.sqrt(np.mean((np.hypot(xc - x0, yc - y0) - r_fit) ** 2)))
  cos_th = float(np.clip((y_wall - y0) / r_fit, -1.0, 1.0))
  theta_num = float(np.rad2deg(np.arccos(cos_th)))
  err_deg = theta_num - theta_set

  sessile_rows.append({
      "theta_set_deg": round(theta_set, 2),
      "theta_num_deg": round(theta_num, 2),
      "err_deg": round(err_deg, 2),
      "R_fit": round(r_fit, 2),
      "rms_fit_LU": round(rms_fit, 3),
      "u_spur": round(u_spur, 6),
  })
  sessile_contours.append((theta_set, theta_num, xc, yc, x0, y0, r_fit))
  print(
      f"{theta_set:12.2f} | {theta_num:15.2f} | {err_deg:+10.2f} |"
      f" {r_fit:10.2f} | {rms_fit:8.3f} | {u_spur:10.6f}"
  )

# --- 3. ANALIZA BENCHMARKU POISEUILLE'A ---
vti_p = latest_vti(os.path.join(BM_DIR, "poiseuille"))


poiseuille_res = {}
if vti_p:
  m = pv.read(vti_p)
  ny, nx = m.dimensions[1] - 1, m.dimensions[0] - 1
  u = m["U"].reshape((ny, nx, 3))
  rho = m["Rho"].reshape((ny, nx))

  print("\n=== BENCHMARK 3: PROFIL POISEUILLE'A (320x40, H=38) ===")
  print("  Kontrola zachowania strumienia wzdloz kanalu:")
  for xs in [0, 1, 100, 250, 318]:
    ux_col = u[1:-1, xs, 0]
    rho_col = rho[1:-1, xs]
    print(
        f"    x={xs:3d}: U_sr={ux_col.mean():.6f} | rho_sr={rho_col.mean():.5f}"
        f" | Q_mas={np.sum(rho_col * ux_col):.5f}"
    )

  x_sample = 250
  ux_sim = u[1:-1, x_sample, 0]
  H = float(ny - 2)
  ys = np.arange(1, ny - 1, dtype=float)
  eta = (ys - 0.5) / H
  u_mean_sim = float(np.mean(ux_sim))

  # Porownanie z parabola dla U_zadane oraz parabola znormalizowana lokalnym U_sr
  ux_ana_target = 6.0 * UC_REF * eta * (1.0 - eta)
  ux_ana_local = 6.0 * u_mean_sim * eta * (1.0 - eta)

  l2_abs_pct = 100.0 * float(
      np.linalg.norm(ux_sim - ux_ana_target) / np.linalg.norm(ux_ana_target)
  )
  l2_shape_pct = 100.0 * float(
      np.linalg.norm(ux_sim - ux_ana_local) / np.linalg.norm(ux_ana_local)
  )
  linf_shape_pct = 100.0 * float(
      np.max(np.abs(ux_sim - ux_ana_local)) / np.max(ux_ana_local)
  )
  # Wyznaczenie efektywnej lepkosci ze spadku cisnienia w strefie rozwinietej
  rho_250 = float(np.mean(rho[1:-1, 250]))
  rho_318 = float(np.mean(rho[1:-1, 318]))
  drho_dx = (rho_318 - rho_250) / (318.0 - 250.0)
  nu_eff = -drho_dx * (H**2) / (36.0 * u_mean_sim)
  nu_err_pct = 100.0 * (nu_eff - 0.025) / 0.025
  print(f"  Lepkosc efektywna (x=250..318): nu_eff={nu_eff:.5f} (blad {nu_err_pct:+.2f}% vs 0.025)")

  poiseuille_res = {
      "H_LU": int(H),
      "x_sample": x_sample,
      "U_mean_target": UC_REF,
      "U_mean_sim": round(u_mean_sim, 6),
      "flux_shift_pct": round(100.0 * (u_mean_sim - UC_REF) / UC_REF, 3),
      "L2_abs_err_pct": round(l2_abs_pct, 3),
      "L2_shape_err_pct": round(l2_shape_pct, 3),
      "Linf_shape_err_pct": round(linf_shape_pct, 3),
  }
  print(
      f"  x={x_sample}: U_sr_sim={u_mean_sim:.6f} ({poiseuille_res['flux_shift_pct']:+.2f}%)"
      f" | Blad ksztaltu L2={l2_shape_pct:.3f}% | Blad calkowity L2={l2_abs_pct:.3f}%"
  )

# Zapis podsumowania JSON
summary = {
    "young_laplace": {
        "sigma_nominal": ST_NOMINAL,
        "sigma_num": round(float(sigma_num), 5),
        "intercept_dP": round(float(intercept), 6),
        "sigma_err_pct": round(float(sigma_err_pct), 2),
        "R2": round(r2_lap, 6),
        "max_u_spur": round(max(r["u_spur"] for r in laplace_rows), 6),
        "cases": laplace_rows,
    },
    "sessile_drop": sessile_rows,
    "poiseuille": poiseuille_res,
}
with open(os.path.join(BM_DIR, "benchmark_summary.json"), "w") as jf:
  json.dump(summary, jf, indent=2)

# Generacja zbiorczego 3-panelowego wykresu walidacyjnego
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# Panel 1: Young-Laplace
r_line = np.linspace(0.0, inv_rs.max() * 1.1, 50)
axes[0].plot(inv_rs, dps, "o", markersize=8, color="#1f77b4", label="LBM (d2q9_csf)")
axes[0].plot(
    r_line, sigma_num * r_line + intercept, "--", color="#d62728",
    label=f"Fit: sigma={sigma_num:.4f}, b={intercept:+.4f}"
)
axes[0].plot(
    r_line, ST_NOMINAL * r_line, ":", color="black",
    label=f"Teoria: sigma={ST_NOMINAL:.3f}"
)
axes[0].set_xlabel("Krzywizna 1 / R [1/LU]")
axes[0].set_ylabel("Skok cisnienia dP [LU]")
axes[0].set_title(f"Prawo Younga-Laplace'a (R2={r2_lap:.5f})")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8.5)

# Panel 2: Sessile Drop (bez gornej sciany)
colors = ["#2ca02c", "#1f77b4", "#d62728"]
for (th_set, th_num, xc, yc, x0, y0, r_fit), col in zip(sessile_contours, colors):
  axes[1].scatter(
      xc - x0, yc - y_wall, s=8, color=col, alpha=0.8,
      label=f"Zad={th_set:.0f}° -> Zmier={th_num:.1f}°"
  )
  ang = np.linspace(0, 2 * np.pi, 300)
  cx_fit = r_fit * np.cos(ang)
  cy_fit = (y0 - y_wall) + r_fit * np.sin(ang)
  above = cy_fit >= 0.0
  axes[1].plot(cx_fit[above], cy_fit[above], "-", color=col, linewidth=1.3, alpha=0.6)
axes[1].axhline(0.0, color="#2b2b2b", linewidth=2.5)
axes[1].set_aspect("equal", adjustable="box")
axes[1].set_xlim(-55, 55)
axes[1].set_ylim(-2, 55)
axes[1].set_xlabel("x - x0 [LU]")
axes[1].set_ylabel("y - y_wall [LU]")
axes[1].set_title("Benchmark kata zwilzania (Sessile Drop)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8, loc="upper right")

# Panel 3: Poiseuille (znormalizowany lokalnym U_sr)
if vti_p:
  axes[2].plot(
      ux_ana_local / u_mean_sim, eta, "-", color="black", linewidth=2.0,
      label="Analityczny Poiseuille"
  )
  axes[2].plot(
      ux_sim / u_mean_sim, eta, "o", markersize=4, color="#d62728",
      label=f"LBM (L2 ksztalt={l2_shape_pct:.2f}%)"
  )
  axes[2].set_xlabel("Bezwymiarowa predkosc u_x / U_sr_lokalne [-]")
  axes[2].set_ylabel("Polozenie w kanale y / H [-]")
  axes[2].set_title(f"Profil Poiseuille'a (shift U_sr: {poiseuille_res['flux_shift_pct']:+.2f}%)")
  axes[2].grid(True, alpha=0.3)
  axes[2].legend(fontsize=8.5)

plt.tight_layout()
out_png = os.path.join(BM_DIR, "validation_benchmarks.png")
plt.savefig(out_png, dpi=180)
plt.close()
print(f"\nZapisano wyniki benchmarkow: {out_png}")