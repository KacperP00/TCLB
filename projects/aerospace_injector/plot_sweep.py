import csv
import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

BASE_DIR = "projects/aerospace_injector/sweep"
json_files = sorted(glob.glob(os.path.join(BASE_DIR, "Uc_*", "summary.json")))

rows = []
for jf in json_files:
  with open(jf, "r") as f:
    rows.append(json.load(f))

rows.sort(key=lambda r: r["U_c"])

if not rows:
  raise SystemExit("Brak plikow summary.json w katalogu sweep/")

# Zapis zbiorczej tabeli CSV
csv_path = os.path.join(BASE_DIR, "sweep_results.csv")
fieldnames = [k for k in rows[0].keys() if k != "last_valid_vti"]
with open(csv_path, "w", newline="") as cf:
  writer = csv.DictWriter(cf, fieldnames=fieldnames)
  writer.writeheader()
  for r in rows:
    writer.writerow({k: r[k] for k in fieldnames})

print("\n=== TABELA ZBIORCZA STUDIUM PARAMETRYCZNEGO (AUTO-PROBES) ===")
header = (
    f"{'U_c':>6s} | {'Ca':>7s} | {'We':>6s} | {'Re':>5s} | {'D21/H':>6s} |"
    f" {'Lb/H':>6s} | {'f[1/10k]':>8s} | {'St':>6s} | {'Ust':>3s} |"
    f" {'AR':>5s} | {'Std%':>5s} | {'Dryf/100':>8s}"
)
print(header)
print("-" * len(header))
for r in rows:
  freq_10k = 10000.0 / r["period_P1_it"] if r["period_P1_it"] > 0 else 0.0
  st_flag = "TAK" if r.get("period_is_steady", True) else "NIE"
  print(
      f"{r['U_c']:6.3f} | {r['Ca']:7.4f} | {r['We']:6.3f} | {r['Re_c']:5.1f} |"
      f" {r['D21_over_H']:6.3f} | {r['Lb_over_H']:6.3f} | {freq_10k:8.2f} |"
      f" {r['Strouhal']:6.4f} | {st_flag:>3s} | {r['AR_mean']:5.2f} |"
      f" {r['D_std_pct']:5.2f} | {r['steady_drift_per_100LU_pct']:+7.2f}%"
  )

cas = np.array([r["Ca"] for r in rows])
wes = np.array([r["We"] for r in rows])
d21_h = np.array([r["D21_over_H"] for r in rows])
lb_h = np.array([r["Lb_over_H"] for r in rows])
st = np.array([r["Strouhal"] for r in rows])
freqs = np.array([
    10000.0 / r["period_P1_it"] if r["period_P1_it"] > 0 else np.nan
    for r in rows
])
steady_mask = np.array([r.get("period_is_steady", True) for r in rows])

# Regresje potegowe na calym zakresie oraz na punktach w pelni ustalonych
b_d, log_a_d = np.polyfit(np.log(cas), np.log(d21_h), 1)
r2_d = np.corrcoef(np.log(cas), np.log(d21_h))[0, 1] ** 2
ca_fit = np.linspace(cas.min(), cas.max(), 100)
d_fit = np.exp(log_a_d) * (ca_fit**b_d)

b_l, log_a_l = np.polyfit(np.log(cas), np.log(lb_h), 1)
r2_l = np.corrcoef(np.log(cas), np.log(lb_h))[0, 1] ** 2
lb_fit = np.exp(log_a_l) * (ca_fit**b_l)

print(f"\nFit (5 pkt) D21/H = {np.exp(log_a_d):.4f} * Ca^({b_d:.3f}) [R^2 = {r2_d:.4f}]")
print(f"Fit (5 pkt) Lb/H  = {np.exp(log_a_l):.4f} * Ca^({b_l:.3f}) [R^2 = {r2_l:.4f}]")

if steady_mask.sum() >= 3 and not np.all(steady_mask):
  b_ds, log_a_ds = np.polyfit(np.log(cas[steady_mask]), np.log(d21_h[steady_mask]), 1)
  r2_ds = np.corrcoef(np.log(cas[steady_mask]), np.log(d21_h[steady_mask]))[0, 1] ** 2
  print(f"Fit (ustalone) D21/H = {np.exp(log_a_ds):.4f} * Ca^({b_ds:.3f}) [R^2 = {r2_ds:.4f}]")

# Wykres 1: Prawa skalowania
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

axes[0].plot(cas, d21_h, "o", markersize=7, color="#1f77b4", label="Dane LBM (d2q9_csf)")
axes[0].plot(
    ca_fit, d_fit, "--", color="#1f77b4", alpha=0.75,
    label=f"Fit: ~Ca^({b_d:.2f}), R2={r2_d:.3f}"
)
axes[0].set_xlabel("Liczba kapilarna Ca [-]")
axes[0].set_ylabel("Bezwymiarowa srednica Sautera D21 / H [-]")
axes[0].set_title("Skalowanie rozmiaru kropli vs Ca")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(cas, lb_h, "s", markersize=7, color="#d62728", label="Dane LBM")
axes[1].plot(
    ca_fit, lb_fit, "--", color="#d62728", alpha=0.75,
    label=f"Fit: ~Ca^({b_l:.2f}), R2={r2_l:.3f}"
)
axes[1].set_xlabel("Liczba kapilarna Ca [-]")
axes[1].set_ylabel("Dlugosc rozpadu strugi Lb / H [-]")
axes[1].set_title("Dlugosc rozpadu vs Ca")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

ax3 = axes[2]
ax3.plot(wes, freqs, "^-", linewidth=2, color="#2ca02c", label="Czestotliwosc f [1/10k it]")
ax3.set_xlabel("Liczba Webera We [-]")
ax3.set_ylabel("Czestotliwosc generacji f [1 / 10k it]", color="#2ca02c")
ax3.tick_params(axis="y", labelcolor="#2ca02c")
ax3.set_title("Czestotliwosc odrywania i St vs We")
ax3.grid(True, alpha=0.3)

ax3_r = ax3.twinx()
ax3_r.plot(wes, st, "D--", linewidth=1.6, color="#ff7f0e", label="Strouhal St = f*H/Uc")
ax3_r.set_ylabel("Liczba Strouhala St [-]", color="#ff7f0e")
ax3_r.tick_params(axis="y", labelcolor="#ff7f0e")
ax3_r.set_ylim(0.0, 0.28)

l1, lb1 = ax3.get_legend_handles_labels()
l2, lb2 = ax3_r.get_legend_handles_labels()
ax3.legend(l1 + l2, lb1 + lb2, loc="lower right", fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "sweep_scaling.png"), dpi=180)
plt.close()

# Wykres 2: 5-panelowy montaz koncowych pol fazowych z pozycja bramek
valid_rows = [r for r in rows if r.get("last_valid_vti") and os.path.exists(r["last_valid_vti"])]
if valid_rows:
  fig, axes = plt.subplots(len(valid_rows), 1, figsize=(13, 1.9 * len(valid_rows)))
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
      ax.axvline(r["probe_1_x"], color="lime", linestyle="--", linewidth=0.9, alpha=0.8)
      ax.axvline(r["probe_2_x"], color="yellow", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_title(
        f"U_c={r['U_c']:.3f} | Ca={r['Ca']:.4f} | We={r['We']:.2f} |"
        f" D21/H={r['D21_over_H']:.2f} | St={r['Strouhal']:.3f}"
    )
    ax.set_yticks([])

  plt.tight_layout()
  plt.savefig(os.path.join(BASE_DIR, "sweep_montage.png"), dpi=180)
  plt.close()

print(f"\nZapisano komplet wynikow w: {BASE_DIR}/")