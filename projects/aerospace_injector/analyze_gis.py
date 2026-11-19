import glob
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

GIS_DIR = "projects/aerospace_injector/gis_fine"
G1_JSON = "projects/aerospace_injector/sweep/Uc_0.028/summary.json"

NOZZLE_RIGHT_X = 138.0
CHANNEL_HEIGHT = 57.0
PROBE_1_X = 666.0
PROBE_2_X = 966.0
PROBE_DIST = PROBE_2_X - PROBE_1_X
UC_FINE = 0.022862
REFINEMENT_RATIO = 1.5
MIN_DROP_CELLS = 110


def extract_iter(path):
  # Odczyt numeru iteracji z nazwy pliku VTI
  m = re.search(r"_(\d+)\.vti$", path)
  return int(m.group(1)) if m else 0


def build_wall_mask_g2(ny, nx):
  # Maska scian geometrii 1920x96 (H=57, D=48)
  mask = np.zeros((ny, nx), dtype=bool)
  mask[-1, :] = True
  mask[:38, :90] = True
  mask[:38, 138:] = True
  return mask


files = sorted(glob.glob(os.path.join(GIS_DIR, "output", "*_VTK_P00_*.vti")))
valid_files = []
breakup_events = []
tracks = {}
next_id = 1
prev_lb = 0.0
first_nan_iter = None

for f in files:
  it = extract_iter(f)
  mesh = pv.read(f)
  ny, nx = mesh.dimensions[1] - 1, mesh.dimensions[0] - 1
  pf = mesh["PhaseField"]

  if np.isnan(pf).any():
    if first_nan_iter is None:
      first_nan_iter = it
    continue

  valid_files.append(f)

  # Maskowanie scian przed progowaniem na granicy faz phi = 0.0
  pf_2d = pf.reshape((ny, nx)).copy()
  pf_2d[build_wall_mask_g2(ny, nx)] = -0.5
  mesh.cell_data["PF_fluid"] = pf_2d.ravel()

  x_cutoff = mesh.bounds[1] - 75.0
  dispersed = mesh.threshold(0.0, scalars="PF_fluid")

  free_drops = []
  lb = 0.0

  if dispersed.n_cells > 0:
    for b in dispersed.split_bodies():
      if b.n_cells < MIN_DROP_CELLS:
        continue
      xmin, xmax, ymin, ymax, _, _ = b.bounds
      d_eq = 2.0 * np.sqrt(b.volume / np.pi)

      if ymin < 15.0:
        lb = max(lb, xmax - NOZZLE_RIGHT_X)
      elif xmax < x_cutoff:
        cx, cy, _ = b.center_of_mass()
        slug_len = xmax - xmin
        ar = slug_len / max(ymax - ymin, 1e-6)
        free_drops.append((cx, cy, d_eq, ar, slug_len))

  if prev_lb - lb > 45.0:
    breakup_events.append((it, prev_lb))
  prev_lb = lb

  # Sledzenie trajektorii kropel
  used = set()
  for cx, cy, d_eq, ar, slug_len in sorted(free_drops, key=lambda x: -x[0]):
    best_id, best_dx = None, 1e9
    for tid, tr in tracks.items():
      if tid in used or not tr["active"]:
        continue
      dx = cx - tr["cx"]
      if -8.0 <= dx <= 140.0 and abs(dx) < best_dx:
        best_dx, best_id = abs(dx), tid

    if best_id is None:
      best_id = next_id
      next_id += 1
      tracks[best_id] = {
          "cx": cx, "it": it, "d_prev": d_eq, "ar_prev": ar,
          "len_prev": slug_len, "active": True, "p1": None, "p2": None,
      }
    else:
      tr = tracks[best_id]
      if tr["p1"] is None and tr["cx"] < PROBE_1_X <= cx:
        w = (PROBE_1_X - tr["cx"]) / max(cx - tr["cx"], 1e-6)
        tr["p1"] = {
            "iter": tr["it"] + w * (it - tr["it"]),
            "d_eq": tr["d_prev"] + w * (d_eq - tr["d_prev"]),
            "ar": tr["ar_prev"] + w * (ar - tr["ar_prev"]),
            "slug_len": tr["len_prev"] + w * (slug_len - tr["len_prev"]),
        }
      if tr["p2"] is None and tr["cx"] < PROBE_2_X <= cx:
        w = (PROBE_2_X - tr["cx"]) / max(cx - tr["cx"], 1e-6)
        tr["p2"] = {
            "iter": tr["it"] + w * (it - tr["it"]),
            "d_eq": tr["d_prev"] + w * (d_eq - tr["d_prev"]),
            "ar": tr["ar_prev"] + w * (ar - tr["ar_prev"]),
            "slug_len": tr["len_prev"] + w * (slug_len - tr["len_prev"]),
        }
      tr.update({"cx": cx, "it": it, "d_prev": d_eq, "ar_prev": ar, "len_prev": slug_len})
    used.add(best_id)

  for tid, tr in tracks.items():
    if tid not in used:
      tr["active"] = False

crossed = sorted(
    [(tid, tr) for tid, tr in tracks.items() if tr["p1"] is not None],
    key=lambda x: x[1]["p1"]["iter"],
)

p1_all, p1_steady, drifts_steady = [], [], []
print(f"\n=== GIS SIATKA GESTA 1920x96 (P1={PROBE_1_X:.0f}, P2={PROBE_2_X:.0f}) ===")
for idx, (tid, tr) in enumerate(crossed, 1):
  p1, p2 = tr["p1"], tr["p2"]
  tag = " [ROZRUCH]" if idx == 1 else ""
  p1_all.append(p1)
  if idx >= 2:
    p1_steady.append(p1)
  if p2 is not None:
    drift_pct = 100.0 * (p2["d_eq"] - p1["d_eq"]) / p1["d_eq"]
    drift_per_10H = drift_pct / (PROBE_DIST / CHANNEL_HEIGHT) * 10.0
    if idx >= 2:
      drifts_steady.append(drift_per_10H)
    print(
        f"  Kropla #{idx}{tag}: P1(it={p1['iter']:6.0f}, D={p1['d_eq']:5.2f} LU,"
        f" AR={p1['ar']:.2f}) -> P2(D={p2['d_eq']:5.2f} LU) |"
        f" dryf={drift_pct:+.2f}% ({drift_per_10H:+.2f}%/10H)"
    )
  else:
    print(
        f"  Kropla #{idx}{tag}: P1(it={p1['iter']:6.0f}, D={p1['d_eq']:5.2f} LU,"
        f" AR={p1['ar']:.2f}) -> [w locie do P2]"
    )

eval_set = p1_steady if p1_steady else p1_all
ds = np.array([p["d_eq"] for p in eval_set]) if eval_set else np.array([0.0])
d21_g2 = float(np.sum(ds**2) / np.sum(ds)) if ds.sum() > 0 else 0.0
d_std_g2 = float(100.0 * np.std(ds) / np.mean(ds)) if ds.mean() > 0 else 0.0
ar_g2 = float(np.mean([p["ar"] for p in eval_set])) if eval_set else 0.0
p1_iters = [p["iter"] for p in p1_steady] if len(p1_steady) >= 2 else [p["iter"] for p in p1_all]
period_g2 = float(np.mean(np.diff(p1_iters))) if len(p1_iters) >= 2 else 0.0
st_g2 = CHANNEL_HEIGHT / (period_g2 * UC_FINE) if period_g2 > 0 else 0.0
lb_vals = [e[1] for e in breakup_events]
lb_steady = lb_vals[1:] if len(lb_vals) >= 2 else lb_vals
lb_h_g2 = float(np.mean(lb_steady)) / CHANNEL_HEIGHT if lb_steady else 0.0
drift_10H_g2 = float(np.mean(drifts_steady)) if drifts_steady else 0.0

g2 = {
    "grid": "1920x96 (H=57)",
    "first_nan_iter": first_nan_iter,
    "steady_drops": len(p1_steady),
    "D21_over_H": round(d21_g2 / CHANNEL_HEIGHT, 4),
    "Lb_over_H": round(lb_h_g2, 4),
    "Strouhal": round(st_g2, 4),
    "AR_mean": round(ar_g2, 3),
    "D_std_pct": round(d_std_g2, 2),
    "drift_per_10H_pct": round(drift_10H_g2, 2),
    "last_valid_vti": valid_files[-1] if valid_files else "",
}

with open(os.path.join(GIS_DIR, "summary_gis.json"), "w") as jf:
  json.dump(g2, jf, indent=2)

if os.path.exists(G1_JSON):
  with open(G1_JSON, "r") as f:
    g1 = json.load(f)
  drift_10H_g1 = g1["steady_drift_per_100LU_pct"] * 3.8
  r2_m1 = REFINEMENT_RATIO**2 - 1.0

  print("\n=== TABELA WERYFIKACJI SIATKOWEJ (GIS: Re=42.56, Ca=0.00778, r=1.5) ===")
  print(
      f"{'Metryka':>16s} | {'G1 (1280x64)':>12s} | {'G2 (1920x96)':>12s} |"
      f" {'Zmiana [%]':>10s} | {'Rich(p=2)':>9s} | {'GCI_fine':>8s}"
  )
  print("-" * 80)
  for name, v1, v2 in [
      ("D21 / H [-]", g1["D21_over_H"], g2["D21_over_H"]),
      ("Lb / H [-]", g1["Lb_over_H"], g2["Lb_over_H"]),
      ("Strouhal [-]", g1["Strouhal"], g2["Strouhal"]),
      ("AR [-]", g1["AR_mean"], g2["AR_mean"]),
  ]:
    e_rel = 100.0 * (v2 - v1) / v2 if v2 != 0 else 0.0
    rich = v2 + (v2 - v1) / r2_m1
    gci = 3.0 * abs(e_rel) / r2_m1
    print(
        f"{name:>16s} | {v1:12.4f} | {v2:12.4f} | {e_rel:+9.2f}% |"
        f" {rich:9.4f} | {gci:7.2f}%"
    )

  # Poprawna zmiana wzgledna bledu (redukcja dryfu wzgledem siatki bazowej G1)
  drift_red = 100.0 * (abs(g2["drift_per_10H_pct"]) - abs(drift_10H_g1)) / abs(drift_10H_g1)
  p_drift = np.log(abs(drift_10H_g1) / max(abs(g2["drift_per_10H_pct"]), 1e-6)) / np.log(REFINEMENT_RATIO)
  print("-" * 80)
  print(
      f"{'Dryf [%/10H]':>16s} | {drift_10H_g1:12.2f} |"
      f" {g2['drift_per_10H_pct']:12.2f} | {drift_red:+9.2f}% |"
      f" {'p=' + f'{p_drift:.2f}':>9s} | {'---':>8s}"
  )
  print(
      f"{'Rozrzut D [%]':>16s} | {g1['D_std_pct']:12.2f} |"
      f" {g2['D_std_pct']:12.2f} | {'---':>10s} | {'---':>9s} | {'---':>8s}"
  )