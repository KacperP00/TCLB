import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

MRT_BM = "projects/aerospace_injector/benchmarks/benchmark_summary.json"
CUM_BM = "projects/aerospace_injector/benchmarks_cumulant/benchmark_summary.json"
MRT_INJ = "projects/aerospace_injector/sweep/Uc_0.028/summary.json"
CUM_INJ = "projects/aerospace_injector/cumulant_cmp/Uc_0.028/summary.json"
LOW_NU_DIR = "projects/aerospace_injector/cumulant_cmp/laplace_R40_lownu/output"
OUT_DIR = "projects/aerospace_injector/cumulant_cmp"

with open(MRT_BM, "r") as f:
  bm_m = json.load(f)
with open(CUM_BM, "r") as f:
  bm_c = json.load(f)
with open(MRT_INJ, "r") as f:
  inj_m = json.load(f)
with open(CUM_INJ, "r") as f:
  inj_c = json.load(f)

# Analyze low-viscosity Laplace R=40 check for Cumulant
low_files = sorted(glob.glob(os.path.join(LOW_NU_DIR, "*_VTK_P00_*.vti")))
low_nu_info = {}
if low_files:
  m = pv.read(low_files[-1])
  ny, nx = m.dimensions[1] - 1, m.dimensions[0] - 1
  pf = m["PhaseField"].reshape((ny, nx))
  rho = m["Rho"].reshape((ny, nx))
  u = m["U"].reshape((ny, nx, 3))
  r_eq = float(np.sqrt(np.sum(pf > 0.0) / np.pi))
  yy, xx = np.mgrid[0:ny, 0:nx]
  rg = np.hypot(xx - (nx / 2.0 - 0.5), yy - (ny / 2.0 - 0.5))
  rin = rho[rg < r_eq - 7.0]
  rout = rho[rg > r_eq + 12.0]
  dp_low = (float(np.mean(rin)) - float(np.mean(rout))) / 3.0
  low_nu_info = {
      "R_eq": round(r_eq, 2),
      "dP": round(dp_low, 6),
      "sigma_eff": round(dp_low * r_eq, 5),
      "std_in": round(float(np.std(rin)), 6),
      "max_u_spur": round(float(np.max(np.linalg.norm(u, axis=2))), 6),
  }
  print("\n=== CUMULANT LOW-VISCOSITY LAPLACE CHECK (R=40, nu=0.025/0.012, 15k it) ===")
  print(
      f"  R_eq={low_nu_info['R_eq']} | dP={low_nu_info['dP']:.6f} |"
      f" sigma_eff={low_nu_info['sigma_eff']:.5f} | std_in={low_nu_info['std_in']:.2e} |"
      f" max|U_sp|={low_nu_info['max_u_spur']:.6f}"
  )

# Print side-by-side comparison table
print("\n=== COLLISION OPERATOR COMPARISON: MRT vs. CUMULANT ===")
print(f"{'Metric':>26s} | {'MRT':>12s} | {'Cumulant':>12s} | {'Diff (Cum-MRT)':>15s}")
print("-" * 74)

rows = [
    ("Laplace sigma_zero [-]", bm_m["young_laplace"]["sigma_zero"], bm_c["young_laplace"]["sigma_zero"]),
    ("Laplace max|U_sp| [LU]", bm_m["young_laplace"]["max_u_spur"], bm_c["young_laplace"]["max_u_spur"]),
    ("Sessile 143 deg [deg]", bm_m["sessile_drop"][-1]["theta_num_deg"], bm_c["sessile_drop"][-1]["theta_num_deg"]),
    ("Poiseuille L2 shape [%]", bm_m["poiseuille"]["L2_shape_err_pct"], bm_c["poiseuille"]["L2_shape_err_pct"]),
    ("Poiseuille nu_eff [-]", bm_m["poiseuille"]["nu_eff"], bm_c["poiseuille"]["nu_eff"]),
    ("Injector D21 / H [-]", inj_m["D21_over_H"], inj_c["D21_over_H"]),
    ("Injector Lb / H [-]", inj_m["Lb_over_H"], inj_c["Lb_over_H"]),
    ("Injector Strouhal [-]", inj_m["Strouhal"], inj_c["Strouhal"]),
    ("Injector AR [-]", inj_m["AR_mean"], inj_c["AR_mean"]),
    ("Injector D_std [%]", inj_m["D_std_pct"], inj_c["D_std_pct"]),
    ("Injector Drift [%/100LU]", inj_m["steady_drift_per_100LU_pct"], inj_c["steady_drift_per_100LU_pct"]),
]

for name, vm, vc in rows:
  rel = 100.0 * (vc - vm) / abs(vm) if vm != 0 else 0.0
  print(f"{name:>26s} | {vm:12.4f} | {vc:12.4f} | {rel:+14.2f}%")

# Save JSON comparison
cmp_out = {
    "benchmarks_mrt": bm_m,
    "benchmarks_cumulant": bm_c,
    "injector_mrt": inj_m,
    "injector_cumulant": inj_c,
    "cumulant_low_nu_laplace_R40": low_nu_info,
}
with open(os.path.join(OUT_DIR, "cumulant_comparison.json"), "w") as jf:
  json.dump(cmp_out, jf, indent=2)

# Visual 3-panel comparison: MRT topology, Cumulant topology, and spurious currents
fig = plt.figure(figsize=(13.5, 6.8))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15], hspace=0.45, wspace=0.28)

ax_m = fig.add_subplot(gs[0, :])
ax_c = fig.add_subplot(gs[1, :])
ax_sp = fig.add_subplot(gs[2, 0])
ax_bar = fig.add_subplot(gs[2, 1])

cmap = plt.cm.coolwarm.copy()
cmap.set_bad(color="#2b2b2b")

for ax, inj, title_prefix in [
    (ax_m, inj_m, "MRT Operator"),
    (ax_c, inj_c, "Cumulant Operator"),
]:
  vti = inj.get("last_valid_vti", "")
  if vti and os.path.exists(vti):
    m = pv.read(vti)
    ny, nx = m.dimensions[1] - 1, m.dimensions[0] - 1
    mask = np.zeros((ny, nx), dtype=bool)
    mask[-1, :] = True; mask[:25, :60] = True; mask[:25, 92:] = True
    pf = np.ma.masked_where(mask, m["PhaseField"].reshape((ny, nx)))
    ax.imshow(pf, origin="lower", cmap=cmap, vmin=-0.5, vmax=0.5)
    ax.contour(pf, levels=[0.0], colors="black", linewidths=0.7)
    ax.set_title(
        f"{title_prefix} ($U_c=0.028$, $t=36000$) | $D_{{21}}/H={inj['D21_over_H']:.3f}$ |"
        f" $L_b/H={inj['Lb_over_H']:.2f}$ | $St={inj['Strouhal']:.4f}$ |"
        f" Drift={inj['steady_drift_per_100LU_pct']:+.2f}%/100LU"
    )
    ax.set_yticks([])

# Spurious currents vs radius (Laplace benchmark)
radii = [c["R_init"] for c in bm_m["young_laplace"]["cases"]]
sp_m = [c["u_spur"] * 1e3 for c in bm_m["young_laplace"]["cases"]]
sp_c = [c["u_spur"] * 1e3 for c in bm_c["young_laplace"]["cases"]]
ax_sp.plot(radii, sp_m, "o-", linewidth=2, color="#1f77b4", label="MRT")
ax_sp.plot(radii, sp_c, "s--", linewidth=2, color="#d62728", label="Cumulant")
ax_sp.set_xlabel("Droplet Radius $R$ [LU]")
ax_sp.set_ylabel("Max Spurious Velocity $\\times 10^{-3}$ [LU]")
ax_sp.set_title("Spurious Currents (Young-Laplace, $\\nu=1/6$)")
ax_sp.grid(True, alpha=0.3)
ax_sp.legend()

# Dimensionless injector metrics bar comparison
labels = ["$D_{21}/H$", "$L_b/(10H)$", "$5 \\cdot St$", "$AR$"]
vals_m = [inj_m["D21_over_H"], inj_m["Lb_over_H"] / 10.0, 5.0 * inj_m["Strouhal"], inj_m["AR_mean"]]
vals_c = [inj_c["D21_over_H"], inj_c["Lb_over_H"] / 10.0, 5.0 * inj_c["Strouhal"], inj_c["AR_mean"]]
x_pos = np.arange(len(labels))
w = 0.35
ax_bar.bar(x_pos - w / 2, vals_m, w, color="#1f77b4", label="MRT")
ax_bar.bar(x_pos + w / 2, vals_c, w, color="#d62728", label="Cumulant")
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(labels)
ax_bar.set_ylabel("Scaled Dimensionless Value [-]")
ax_bar.set_title("T-Junction Baseline Metrics ($U_c = 0.028$)")
ax_bar.grid(True, axis="y", alpha=0.3)
ax_bar.legend()

out_png = os.path.join(OUT_DIR, "mrt_vs_cumulant.png")
plt.savefig(out_png, dpi=180)
plt.close()
print(f"\nSaved operator comparison figure: {out_png}")