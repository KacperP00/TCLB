import glob
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

NOZZLE_RIGHT_X = 92.0
CHANNEL_HEIGHT = 38.0
PROBE_1_X = 300.0
PROBE_2_X = 540.0

files = sorted(glob.glob("projects/aerospace_injector/output/*_VTK_P00_*.vti"))


def extract_iter(path):
  # Odczyt numeru iteracji z nazwy pliku VTI
  m = re.search(r"_(\d+)\.vti$", path)
  return int(m.group(1)) if m else 0


def build_wall_mask(ny, nx):
  # Maska scian geometrii T-Junction 768x64 (H=38, D=32)
  mask = np.zeros((ny, nx), dtype=bool)
  mask[-1, :] = True
  mask[:25, :60] = True
  mask[:25, 92:] = True
  return mask


valid_files = []
iters = []
lb_series = []
breakup_events = []

# Slownik sledzonych kropel: id -> stan poprzedni i pomiary z bramek
tracks = {}
next_id = 1
prev_lb = 0.0
first_nan_iter = None

for f in files:
  it = extract_iter(f)
  mesh = pv.read(f)
  dims = mesh.dimensions
  ny, nx = dims[1] - 1, dims[0] - 1
  pf = mesh["PhaseField"]
  nan_count = int(np.isnan(pf).sum())

  if nan_count > 0:
    if first_nan_iter is None:
      first_nan_iter = it
    print(f"iter {it:6d} | NaN: {nan_count} (przerwano analize klatki)")
    continue

  valid_files.append(f)

  # Wyzerowanie pola na scianach przed progowaniem na fizycznej granicy phi = 0.0
  wall_mask = build_wall_mask(ny, nx)
  pf_2d = pf.reshape((ny, nx)).copy()
  pf_2d[wall_mask] = -0.5
  mesh.cell_data["PF_fluid"] = pf_2d.ravel()

  x_cutoff = mesh.bounds[1] - 50.0
  dispersed = mesh.threshold(0.0, scalars="PF_fluid")

  free_drops = []
  lb = 0.0

  if dispersed.n_cells > 0:
    for b in dispersed.split_bodies():
      if b.n_cells <= 10:
        continue
      xmin, xmax, ymin, ymax, _, _ = b.bounds
      d_eq = 2.0 * np.sqrt(b.volume / np.pi)

      if ymin < 10.0:
        lb = max(lb, xmax - NOZZLE_RIGHT_X)
      elif xmax < x_cutoff:
        cx, cy, _ = b.center_of_mass()
        slug_len = xmax - xmin
        ar = slug_len / max(ymax - ymin, 1e-6)
        free_drops.append((cx, cy, d_eq, ar, slug_len))

  # Detekcja momentu oderwania kropli na podstawie spadku dlugosci strugi
  if prev_lb - lb > 30.0:
    breakup_events.append((it, prev_lb))
  prev_lb = lb
  iters.append(it)
  lb_series.append(lb)

  # Dopasowanie wykrytych kropel do aktywnych sciezek (od prawej do lewej)
  used_tracks = set()
  for cx, cy, d_eq, ar, slug_len in sorted(free_drops, key=lambda x: -x[0]):
    best_id = None
    best_dx = 1e9
    for tid, tr in tracks.items():
      if tid in used_tracks or not tr["active"]:
        continue
      dx = cx - tr["cx"]
      if -5.0 <= dx <= 65.0 and abs(dx) < best_dx:
        best_dx = abs(dx)
        best_id = tid

    if best_id is None:
      best_id = next_id
      next_id += 1
      tracks[best_id] = {
          "cx": cx,
          "it": it,
          "d_prev": d_eq,
          "ar_prev": ar,
          "len_prev": slug_len,
          "active": True,
          "p1": None,
          "p2": None,
      }
    else:
      tr = tracks[best_id]
      prev_cx, prev_it, prev_d = tr["cx"], tr["it"], tr["d_prev"]
      prev_ar, prev_len = tr["ar_prev"], tr["len_prev"]

      # Interpolacja liniowa przejscia srodka masy przez bramke 1
      if tr["p1"] is None and prev_cx < PROBE_1_X <= cx:
        w = (PROBE_1_X - prev_cx) / max(cx - prev_cx, 1e-6)
        tr["p1"] = {
            "iter": prev_it + w * (it - prev_it),
            "d_eq": prev_d + w * (d_eq - prev_d),
            "ar": prev_ar + w * (ar - prev_ar),
            "slug_len": prev_len + w * (slug_len - prev_len),
        }

      # Interpolacja liniowa przejscia srodka masy przez bramke 2
      if tr["p2"] is None and prev_cx < PROBE_2_X <= cx:
        w = (PROBE_2_X - prev_cx) / max(cx - prev_cx, 1e-6)
        tr["p2"] = {
            "iter": prev_it + w * (it - prev_it),
            "d_eq": prev_d + w * (d_eq - prev_d),
            "ar": prev_ar + w * (ar - prev_ar),
            "slug_len": prev_len + w * (slug_len - prev_len),
        }

      tr["cx"] = cx
      tr["it"] = it
      tr["d_prev"] = d_eq
      tr["ar_prev"] = ar
      tr["len_prev"] = slug_len

    used_tracks.add(best_id)

  for tid, tr in tracks.items():
    if tid not in used_tracks:
      tr["active"] = False

# Zestawienie wynikow dla kazdej zarejestrowanej kropli
print("\n=== POMIAR NA BRAMKACH OPTYCZNYCH (x1=300, x2=540) ===")
p1_all = []
p1_steady = []
drifts = []
vels = []

for tid in sorted(tracks.keys()):
  tr = tracks[tid]
  p1, p2 = tr["p1"], tr["p2"]
  if p1 is None:
    continue
  p1_all.append(p1)
  if tid >= 2:
    p1_steady.append(p1)

  if p2 is not None:
    drift_pct = 100.0 * (p2["d_eq"] - p1["d_eq"]) / p1["d_eq"]
    vel = (PROBE_2_X - PROBE_1_X) / max(p2["iter"] - p1["iter"], 1e-6)
    drifts.append(drift_pct)
    vels.append(vel)
    tag = " [ROZRUCH]" if tid == 1 else ""
    print(
        f"  Kropla #{tid}{tag}: P1(it={p1['iter']:6.0f}, D={p1['d_eq']:5.2f} LU,"
        f" AR={p1['ar']:.2f}) -> P2(D={p2['d_eq']:5.2f} LU) |"
        f" dryf={drift_pct:+.2f}% | U={vel:.4f} LU/it"
    )
  else:
    tag = " [ROZRUCH]" if tid == 1 else ""
    print(
        f"  Kropla #{tid}{tag}: P1(it={p1['iter']:6.0f}, D={p1['d_eq']:5.2f} LU,"
        f" AR={p1['ar']:.2f}) -> [nie dotarla do P2]"
    )

# Metryki stanu ustalonego (od kropli #2; fallback na wszystkie przy 1 kropli)
eval_set = p1_steady if p1_steady else p1_all
periods = (
    np.diff([e[0] for e in breakup_events])
    if len(breakup_events) >= 2
    else np.array([])
)

if eval_set:
  ds = np.array([p["d_eq"] for p in eval_set])
  ars = np.array([p["ar"] for p in eval_set])
  lens = np.array([p["slug_len"] for p in eval_set])
  d10 = float(np.mean(ds))
  d21 = float(np.sum(ds**2) / np.sum(ds))
  d_std_pct = float(100.0 * np.std(ds) / d10)
  ar_mean = float(np.mean(ars))
  len_mean = float(np.mean(lens))
else:
  d10 = d21 = d_std_pct = ar_mean = len_mean = 0.0

summary = {
    "first_nan_iter": first_nan_iter,
    "n_breakups": len(breakup_events),
    "steady_drops_measured": len(p1_steady),
    "mean_period_it": round(float(np.mean(periods)), 1) if len(periods) else 0.0,
    "mean_lb_max_LU": round(float(np.mean([e[1] for e in breakup_events])), 2)
    if breakup_events
    else 0.0,
    "D10_steady_LU": round(d10, 2),
    "D21_steady_LU": round(d21, 2),
    "D21_over_H": round(d21 / CHANNEL_HEIGHT, 3),
    "L_slug_over_H": round(len_mean / CHANNEL_HEIGHT, 3),
    "AR_mean": round(ar_mean, 2),
    "D_std_pct": round(d_std_pct, 2),
    "mean_drift_240LU_pct": round(float(np.mean(drifts)), 2) if drifts else 0.0,
    "mean_U_drop": round(float(np.mean(vels)), 4) if vels else 0.0,
}

print("\n=== RAPORT STANU USTALONEGO ===")
for k, v in summary.items():
  print(f"  {k}: {v}")

with open("projects/aerospace_injector/last_summary.json", "w") as jf:
  json.dump(summary, jf, indent=2)

# Wizualizacja 3 klatek z zaznaczonymi bramkami optycznymi P1 i P2
if valid_files:
  idxs = np.unique(
      np.linspace(0, len(valid_files) - 1, min(3, len(valid_files)), dtype=int)
  )
  selected = [valid_files[i] for i in idxs]

  fig, axes = plt.subplots(len(selected), 1, figsize=(12, 2.3 * len(selected)))
  if len(selected) == 1:
    axes = [axes]

  cmap = plt.cm.coolwarm.copy()
  cmap.set_bad(color="#2b2b2b")

  for ax, f in zip(axes, selected):
    m = pv.read(f)
    d = m.dimensions
    ny_f, nx_f = d[1] - 1, d[0] - 1
    pf_masked = np.ma.masked_where(
        build_wall_mask(ny_f, nx_f), m["PhaseField"].reshape((ny_f, nx_f))
    )
    im = ax.imshow(pf_masked, origin="lower", cmap=cmap, vmin=-0.5, vmax=0.5)
    ax.contour(pf_masked, levels=[0.0], colors="black", linewidths=0.8)
    ax.axvline(PROBE_1_X, color="lime", linestyle="--", linewidth=1.1)
    ax.axvline(PROBE_2_X, color="yellow", linestyle="--", linewidth=1.1)
    ax.set_title(f"{os.path.basename(f)} (zielona=P1, zolta=P2)")
    plt.colorbar(im, ax=ax, label="PhaseField")

  plt.tight_layout()
  plt.savefig("projects/aerospace_injector/diagnostyka.png", dpi=150)

# Wykres telemetrii dlugosci rozpadu Lb(t)
if iters:
  plt.figure(figsize=(10, 4))
  plt.plot(iters, lb_series, "-o", markersize=3, label="Dlugosc strugi Lb [LU]")
  for ev_it, _ in breakup_events:
    plt.axvline(ev_it, color="red", linestyle=":", alpha=0.7)
  plt.xlabel("Iteracja [LU]")
  plt.ylabel("Lb [LU]")
  plt.title("Cykl formowania i rozpadu strugi (czerwone linie = oderwanie kropli)")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig("projects/aerospace_injector/telemetria.png", dpi=150)