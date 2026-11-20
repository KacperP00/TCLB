import csv
import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

BASE_DIR = "projects/aerospace_injector/sweep"
H_CH = 38.0
D_NOZ = 32.0
U_D = 0.008

json_files = sorted(glob.glob(os.path.join(BASE_DIR, "Uc_*", "summary.json")))
rows = []
for jf in json_files:
  with open(jf, "r") as f:
    rows.append(json.load(f))

rows.sort(key=lambda r: r["U_c"])
if not rows:
  raise SystemExit("No summary.json files found in sweep/")

# Export aggregated CSV table
csv_path = os.path.join(BASE_DIR, "sweep_results.csv")
fieldnames = [k for k in rows[0].keys() if k != "last_valid_vti"]
with open(csv_path, "w", newline="") as cf:
  writer = csv.DictWriter(cf, fieldnames=fieldnames)
  writer.writeheader()
  for r in rows:
    writer.writerow({k: r[k] for k in fieldnames})

print("\n=== PARAMETRIC SWEEP SUMMARY TABLE (AUTO-PROBES) ===")
header = (
    f"{'U_c':>6s} | {'Ca':>7s} | {'We':>6s} | {'Re':>5s} | {'Qd/Qc':>5s} |"
    f" {'D21/H':>6s} | {'Lb/H':>6s} | {'f[1/10k]':>8s} | {'St':>6s} |"
    f" {'Stdy':>4s} | {'AR':>5s} | {'Std%':>5s} | {'Drift/100':>9s}"
)
print(header)
print("-" * len(header))
for r in rows:
  qd_qc = (U_D * D_NOZ) / (r["U_c"] * H_CH)
  freq_10k = 10000.0 / r["period_P1_it"] if r["period_P1_it"] > 0 else 0.0
  st_flag = "YES" if r.get("period_is_steady", True) else "NO"
  print(
      f"{r['U_c']:6.3f} | {r['Ca']:7.4f} | {r['We']:6.3f} | {r['Re_c']:5.1f} |"
      f" {qd_qc:5.2f} | {r['D21_over_H']:6.3f} | {r['Lb_over_H']:6.3f} |"
      f" {freq_10k:8.2f} | {r['Strouhal']:6.4f} | {st_flag:>4s} |"
      f" {r['AR_mean']:5.2f} | {r['D_std_pct']:5.2f} |"
      f" {r['steady_drift_per_100LU_pct']:+8.2f}%"
  )

ucs = np.array([r["U_c"] for r in rows])
cas = np.array([r["Ca"] for r in rows])
wes = np.array([r["We"] for r in rows])
qd_qcs = (U_D * D_NOZ) / (ucs * H_CH)
d21_h = np.array([r["D21_over_H"] for r in rows])
l_slug_h = np.array([r["L_slug_over_H"] for r in rows])
lb_h = np.array([r["Lb_over_H"] for r in rows])
st = np.array([r["Strouhal"] for r in rows])
freqs = np.array([
    10000.0 / r["period_P1_it"] if r["period_P1_it"] > 0 else np.nan
    for r in rows
])
steady_mask = np.array([r.get("period_is_steady", True) for r in rows])

# Power-law fits across all cases and steady dripping regime
b_d, log_a_d = np.polyfit(np.log(cas), np.log(d21_h), 1)
r2_d = np.corrcoef(np.log(cas), np.log(d21_h))[0, 1] ** 2
ca_fit = np.linspace(cas.min(), cas.max(), 100)
d_fit = np.exp(log_a_d) * (ca_fit**b_d)

b_l, log_a_l = np.polyfit(np.log(cas), np.log(lb_h), 1)
r2_l = np.corrcoef(np.log(cas), np.log(lb_h))[0, 1] ** 2
lb_fit = np.exp(log_a_l) * (ca_fit**b_l)

print(f"\nPower-law fit (all 5 pts):  D21/H = {np.exp(log_a_d):.4f} * Ca^({b_d:.3f}) [R^2 = {r2_d:.4f}]")
print(f"Power-law fit (all 5 pts):  Lb/H  = {np.exp(log_a_l):.4f} * Ca^({b_l:.3f}) [R^2 = {r2_l:.4f}]")

b_ds, log_a_ds, r2_ds = b_d, log_a_d, r2_d
if steady_mask.sum() >= 3 and not np.all(steady_mask):
  b_ds, log_a_ds = np.polyfit(np.log(cas[steady_mask]), np.log(d21_h[steady_mask]), 1)
  r2_ds = np.corrcoef(np.log(cas[steady_mask]), np.log(d21_h[steady_mask]))[0, 1] ** 2
  print(f"Power-law fit (steady pts): D21/H = {np.exp(log_a_ds):.4f} * Ca^({b_ds:.3f}) [R^2 = {r2_ds:.4f}]")

# Figure 1: Three-panel dimensionless scaling laws
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

axes[0].plot(cas, d21_h, "o", markersize=7, color="#1f77b4", label="LBM data (d2q9_csf)")
axes[0].plot(
    ca_fit, d_fit, "--", color="#1f77b4", alpha=0.75,
    label=f"Fit: $\\propto Ca^{{{b_d:.2f}}}$ ($R^2={r2_d:.3f}$)"
)
axes[0].set_xlabel("Capillary Number $Ca$ [-]")
axes[0].set_ylabel("Dimensionless Sauter Diameter $D_{21} / H$ [-]")
axes[0].set_title("Droplet Size Scaling vs. $Ca$")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(cas, lb_h, "s", markersize=7, color="#d62728", label="LBM data")
axes[1].plot(
    ca_fit, lb_fit, "--", color="#d62728", alpha=0.75,
    label=f"Fit: $\\propto Ca^{{{b_l:.2f}}}$ ($R^2={r2_l:.3f}$)"
)
axes[1].set_xlabel("Capillary Number $Ca$ [-]")
axes[1].set_ylabel("Dimensionless Breakup Length $L_b / H$ [-]")
axes[1].set_title("Jet Breakup Length vs. $Ca$")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

ax3 = axes[2]
ax3.plot(wes, freqs, "^-", linewidth=2, color="#2ca02c", label="Shedding frequency $f$ [1/$10^4$ it]")
ax3.set_xlabel("Weber Number $We$ [-]")
ax3.set_ylabel("Shedding Frequency $f$ [1 / $10^4$ it]", color="#2ca02c")
ax3.tick_params(axis="y", labelcolor="#2ca02c")
ax3.set_title("Shedding Frequency & Strouhal Number vs. $We$")
ax3.grid(True, alpha=0.3)

ax3_r = ax3.twinx()
ax3_r.plot(wes, st, "D--", linewidth=1.6, color="#ff7f0e", label="Strouhal $St = f H / U_c$")
ax3_r.set_ylabel("Strouhal Number $St$ [-]", color="#ff7f0e")
ax3_r.tick_params(axis="y", labelcolor="#ff7f0e")
ax3_r.set_ylim(0.0, 0.28)

l1, lb1 = ax3.get_legend_handles_labels()
l2, lb2 = ax3_r.get_legend_handles_labels()
ax3.legend(l1 + l2, lb1 + lb2, loc="lower right", fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "sweep_scaling.png"), dpi=180)
plt.close()

# Figure 2: Log-log scaling and Garstecki squeezing-to-dripping transition analysis
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

axes[0].loglog(cas[steady_mask], d21_h[steady_mask], "o", markersize=8, color="#1f77b4", label="Steady dripping ($Ca \\geq 0.0064$)")
if np.any(~steady_mask):
  axes[0].loglog(cas[~steady_mask], d21_h[~steady_mask], "s", markersize=8, color="#ff7f0e", label="Transitional slugging ($Ca = 0.0050$)")
ca_steady_fit = np.linspace(cas[steady_mask].min(), cas[steady_mask].max(), 50)
axes[0].loglog(
    ca_steady_fit, np.exp(log_a_ds) * (ca_steady_fit**b_ds), "--", color="#1f77b4",
    label=f"Steady fit: $D_{{21}}/H = {np.exp(log_a_ds):.4f}\\,Ca^{{{b_ds:.3f}}}$ ($R^2={r2_ds:.4f}$)"
)
axes[0].set_xlabel("Capillary Number $Ca$ [-]")
axes[0].set_ylabel("Dimensionless Sauter Diameter $D_{21} / H$ [-]")
axes[0].set_title("Log-Log Power-Law Scaling in Dripping Regime")
axes[0].grid(True, which="both", alpha=0.3)
axes[0].legend(fontsize=8.5)

# Garstecki (2006) squeezing scaling vs LBM droplet axial length
q_line = np.linspace(0.14, 0.40, 50)
axes[1].plot(qd_qcs, l_slug_h, "o-", linewidth=2, markersize=7, color="#d62728", label="LBM axial length $L_{\\mathrm{slug}} / H$")
axes[1].plot(q_line, 1.0 + 0.70 * q_line, "--", color="black", linewidth=1.4, label="Garstecki squeezing: $1 + 0.70\\,(Q_d/Q_c)$")
axes[1].plot(q_line, 1.0 + 1.00 * q_line, ":", color="gray", linewidth=1.4, label="Garstecki squeezing: $1 + 1.00\\,(Q_d/Q_c)$")
axes[1].axhline(1.0, color="#2ca02c", linestyle="-.", linewidth=1.0, alpha=0.8, label="Confinement threshold $L/H = 1$")
axes[1].set_xlabel("Flow Rate Ratio $Q_d / Q_c$ [-]")
axes[1].set_ylabel("Normalized Droplet Length $L_{\\mathrm{slug}} / H$ [-]")
axes[1].set_title("Squeezing-to-Dripping Regime Transition")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "sweep_regime_analysis.png"), dpi=180)
plt.close()

# Figure 3: Five-panel phase-field montage with optical probe locations
valid_rows = [r for r in rows if r.get("last_valid_vti") and os.path.exists(r["last_valid_vti"])]
if valid_rows:
  fig, axes = plt.subplots(len(valid_rows), 1, figsize=(13, 1.95 * len(valid_rows)))
  if len(valid_rows) == 1:
    axes = [axes]

  cmap = plt.cm.coolwarm.copy()
  cmap.set_bad(color="#2b2b2b")

  for ax, r in zip(axes, valid_rows):
    m = pv.read(r["last_valid_vti"])
    ny, nx = m.dimensions[1] - 1, m.dimensions[0] - 1
    mask = np.zeros((ny, nx), dtype=bool)
    mask[-1, :] = True
    mask[:25, :60] = True
    mask[:25, 92:] = True
    pf = np.ma.masked_where(mask, m["PhaseField"].reshape((ny, nx)))

    ax.imshow(pf, origin="lower", cmap=cmap, vmin=-0.5, vmax=0.5)
    ax.contour(pf, levels=[0.0], colors="black", linewidths=0.7)
    if "probe_1_x" in r:
      ax.axvline(r["probe_1_x"], color="lime", linestyle="--", linewidth=0.9, alpha=0.85)
      ax.axvline(r["probe_2_x"], color="yellow", linestyle="--", linewidth=0.9, alpha=0.85)
    ax.set_title(
        f"$U_c = {r['U_c']:.3f}$ | $Ca = {r['Ca']:.4f}$ | $We = {r['We']:.2f}$ |"
        f" $D_{{21}}/H = {r['D21_over_H']:.2f}$ | $St = {r['Strouhal']:.3f}$"
    )
    ax.set_yticks([])

  axes[-1].set_xlabel("Axial Position $x$ [LU] (dashed green: P1, yellow: P2)")
  plt.tight_layout()
  plt.savefig(os.path.join(BASE_DIR, "sweep_montage.png"), dpi=180)
  plt.close()

print(f"\nSaved all English sweep outputs to: {BASE_DIR}/")