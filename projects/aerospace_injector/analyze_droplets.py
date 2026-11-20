import argparse
import glob
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

NOZZLE_RIGHT_X = 92.0
CHANNEL_HEIGHT = 38.0
MIN_DROP_CELLS = 50


def extract_iter(path):
  # Parse iteration number from VTI filename
  m = re.search(r"_(\d+)\.vti$", path)
  return int(m.group(1)) if m else 0


def build_wall_mask(ny, nx):
  # Construct boolean wall mask for T-junction geometry (H=38, D=32)
  mask = np.zeros((ny, nx), dtype=bool)
  mask[-1, :] = True
  mask[:25, :60] = True
  mask[:25, 92:] = True
  return mask


def scan_frames(files):
  # Read VTI series and extract jet breakup length and detached droplets
  valid_files = []
  frame_data = []
  breakup_events = []
  prev_lb = 0.0
  first_nan_iter = None

  for f in files:
    it = extract_iter(f)
    mesh = pv.read(f)
    ny, nx = mesh.dimensions[1] - 1, mesh.dimensions[0] - 1
    pf = mesh["PhaseField"]
    nan_count = int(np.isnan(pf).sum())

    if nan_count > 0:
      if first_nan_iter is None:
        first_nan_iter = it
      print(f"iter {it:6d} | NaN: {nan_count} (frame skipped)")
      continue

    valid_files.append(f)

    # Mask solid walls before thresholding at physical interface phi = 0.0
    pf_2d = pf.reshape((ny, nx)).copy()
    pf_2d[build_wall_mask(ny, nx)] = -0.5
    mesh.cell_data["PF_fluid"] = pf_2d.ravel()

    x_cutoff = mesh.bounds[1] - 50.0
    dispersed = mesh.threshold(0.0, scalars="PF_fluid")

    free_drops = []
    lb = 0.0

    if dispersed.n_cells > 0:
      for b in dispersed.split_bodies():
        if b.n_cells < MIN_DROP_CELLS:
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

    # Detect pinch-off from sudden retraction of attached jet length
    if prev_lb - lb > 30.0:
      breakup_events.append((it, prev_lb))
    prev_lb = lb
    frame_data.append((it, lb, free_drops))

  return valid_files, frame_data, breakup_events, first_nan_iter


def track_on_probes(frame_data, probe_1_x, probe_2_x):
  # Track droplets across frames and interpolate optical probe crossings
  tracks = {}
  next_id = 1

  for it, _, free_drops in frame_data:
    used = set()
    for cx, cy, d_eq, ar, slug_len in sorted(free_drops, key=lambda x: -x[0]):
      best_id = None
      best_dx = 1e9
      for tid, tr in tracks.items():
        if tid in used or not tr["active"]:
          continue
        dx = cx - tr["cx"]
        if -5.0 <= dx <= 90.0 and abs(dx) < best_dx:
          best_dx = abs(dx)
          best_id = tid

      if best_id is None:
        best_id = next_id
        next_id += 1
        tracks[best_id] = {
            "cx": cx,
            "x_birth": cx,
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

        if tr["p1"] is None and prev_cx < probe_1_x <= cx:
          w = (probe_1_x - prev_cx) / max(cx - prev_cx, 1e-6)
          tr["p1"] = {
              "iter": prev_it + w * (it - prev_it),
              "d_eq": prev_d + w * (d_eq - prev_d),
              "ar": prev_ar + w * (ar - prev_ar),
              "slug_len": prev_len + w * (slug_len - prev_len),
          }

        if tr["p2"] is None and prev_cx < probe_2_x <= cx:
          w = (probe_2_x - prev_cx) / max(cx - prev_cx, 1e-6)
          tr["p2"] = {
              "iter": prev_it + w * (it - prev_it),
              "d_eq": prev_d + w * (d_eq - prev_d),
              "ar": prev_ar + w * (ar - prev_ar),
              "slug_len": prev_len + w * (slug_len - prev_len),
          }

        tr.update({
            "cx": cx,
            "it": it,
            "d_prev": d_eq,
            "ar_prev": ar,
            "len_prev": slug_len,
        })

      used.add(best_id)

    for tid, tr in tracks.items():
      if tid not in used:
        tr["active"] = False

  return tracks


def main():
  parser = argparse.ArgumentParser(description="Droplet analysis from TCLB VTI output")
  parser.add_argument("--input-dir", default="projects/aerospace_injector/output")
  parser.add_argument("--out-json", default="projects/aerospace_injector/last_summary.json")
  parser.add_argument("--plot-dir", default="projects/aerospace_injector")
  parser.add_argument("--p1", type=float, default=480.0)
  parser.add_argument("--p2", type=float, default=680.0)
  parser.add_argument("--auto-probes", action="store_true")
  parser.add_argument("--probe-offset", type=float, default=110.0)
  parser.add_argument("--probe-dist", type=float, default=200.0)
  parser.add_argument("--uc", type=float, default=0.028)
  parser.add_argument("--sigma", type=float, default=0.090)
  parser.add_argument("--no-plots", action="store_true")
  args = parser.parse_args()

  files = sorted(glob.glob(os.path.join(args.input_dir, "*_VTK_P00_*.vti")))
  valid_files, frame_data, breakup_events, first_nan_iter = scan_frames(files)

  # Steady-state breakup length (excluding startup pinch-off when n >= 2)
  lb_vals = [e[1] for e in breakup_events]
  lb_steady = lb_vals[1:] if len(lb_vals) >= 2 else lb_vals
  mean_lb_steady = float(np.mean(lb_steady)) if lb_steady else 0.0

  # Set relative or fixed optical probe coordinates
  if args.auto_probes and mean_lb_steady > 0.0:
    probe_1_x = round(NOZZLE_RIGHT_X + mean_lb_steady + args.probe_offset)
    probe_2_x = probe_1_x + args.probe_dist
  else:
    probe_1_x = args.p1
    probe_2_x = args.p2
  probe_dist = probe_2_x - probe_1_x

  tracks = track_on_probes(frame_data, probe_1_x, probe_2_x)

  crossed_p1 = sorted(
      [(tid, tr) for tid, tr in tracks.items() if tr["p1"] is not None],
      key=lambda item: item[1]["p1"]["iter"],
  )

  print(
      f"\n=== OPTICAL PROBE MEASUREMENTS (P1={probe_1_x:.0f},"
      f" P2={probe_2_x:.0f}, auto={args.auto_probes}) ==="
  )
  p1_all, p1_steady = [], []
  drifts_all, drifts_steady = [], []
  vels_all, vels_steady = [], []

  for order_idx, (tid, tr) in enumerate(crossed_p1, 1):
    p1, p2 = tr["p1"], tr["p2"]
    is_startup = order_idx == 1
    tag = " [STARTUP]" if is_startup else ""

    p1_all.append(p1)
    if not is_startup:
      p1_steady.append(p1)

    if p2 is not None:
      drift_pct = 100.0 * (p2["d_eq"] - p1["d_eq"]) / p1["d_eq"]
      drift_per_100 = drift_pct / (probe_dist / 100.0)
      vel = probe_dist / max(p2["iter"] - p1["iter"], 1e-6)
      drifts_all.append(drift_pct)
      vels_all.append(vel)
      if not is_startup:
        drifts_steady.append(drift_pct)
        vels_steady.append(vel)
      print(
          f"  Droplet #{order_idx}{tag}: P1(it={p1['iter']:6.0f},"
          f" D={p1['d_eq']:5.2f} LU, AR={p1['ar']:.2f}) ->"
          f" P2(D={p2['d_eq']:5.2f} LU, AR={p2['ar']:.2f}) |"
          f" drift={drift_pct:+.2f}% ({drift_per_100:+.2f}%/100LU) |"
          f" U={vel:.4f} LU/it"
      )
    else:
      print(
          f"  Droplet #{order_idx}{tag}: P1(it={p1['iter']:6.0f},"
          f" D={p1['d_eq']:5.2f} LU, AR={p1['ar']:.2f}) -> [in flight to P2,"
          f" x_last={tr['cx']:.1f} LU]"
      )

  eval_set = p1_steady if p1_steady else p1_all
  eval_drifts = drifts_steady if drifts_steady else drifts_all
  eval_vels = vels_steady if vels_steady else vels_all

  # Compute shedding period and mark whether it uses steady-state droplets only
  p1_iters_steady = [p["iter"] for p in p1_steady]
  p1_iters_all = [p["iter"] for p in p1_all]
  if len(p1_iters_steady) >= 2:
    period_p1 = float(np.mean(np.diff(p1_iters_steady)))
    period_is_steady = True
  elif len(p1_iters_all) >= 2:
    period_p1 = float(np.mean(np.diff(p1_iters_all)))
    period_is_steady = False
  else:
    period_p1 = 0.0
    period_is_steady = False

  strouhal = (
      CHANNEL_HEIGHT / (period_p1 * args.uc)
      if (period_p1 > 0.0 and args.uc > 0.0)
      else 0.0
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

  mean_drift = float(np.mean(eval_drifts)) if eval_drifts else 0.0
  nu_c = 0.025
  ca = nu_c * args.uc / args.sigma
  we = (args.uc**2) * CHANNEL_HEIGHT / args.sigma
  re_c = args.uc * CHANNEL_HEIGHT / nu_c

  summary = {
      "U_c": args.uc,
      "sigma_used": round(args.sigma, 6),
      "Ca": round(ca, 5),
      "We": round(we, 4),
      "Re_c": round(re_c, 2),
      "first_nan_iter": first_nan_iter,
      "probe_1_x": probe_1_x,
      "probe_2_x": probe_2_x,
      "n_breakups": len(breakup_events),
      "steady_drops_measured": len(p1_steady),
      "period_is_steady": period_is_steady,
      "period_P1_it": round(period_p1, 1),
      "Strouhal": round(strouhal, 4),
      "mean_lb_steady_LU": round(mean_lb_steady, 2),
      "Lb_over_H": round(mean_lb_steady / CHANNEL_HEIGHT, 3),
      "D10_steady_LU": round(d10, 2),
      "D21_steady_LU": round(d21, 2),
      "D21_over_H": round(d21 / CHANNEL_HEIGHT, 3),
      "L_slug_over_H": round(len_mean / CHANNEL_HEIGHT, 3),
      "AR_mean": round(ar_mean, 2),
      "D_std_pct": round(d_std_pct, 2),
      "steady_drift_per_100LU_pct": round(mean_drift / (probe_dist / 100.0), 2),
      "mean_U_drop": round(float(np.mean(eval_vels)), 4) if eval_vels else 0.0,
      "last_valid_vti": valid_files[-1] if valid_files else "",
  }

  print("\n=== STEADY-STATE SUMMARY ===")
  for k, v in summary.items():
    print(f"  {k}: {v}")

  os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
  with open(args.out_json, "w") as jf:
    json.dump(summary, jf, indent=2)

  if not args.no_plots and valid_files:
    os.makedirs(args.plot_dir, exist_ok=True)
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
      ny_f, nx_f = m.dimensions[1] - 1, m.dimensions[0] - 1
      pf_masked = np.ma.masked_where(
          build_wall_mask(ny_f, nx_f), m["PhaseField"].reshape((ny_f, nx_f))
      )
      im = ax.imshow(pf_masked, origin="lower", cmap=cmap, vmin=-0.5, vmax=0.5)
      ax.contour(pf_masked, levels=[0.0], colors="black", linewidths=0.8)
      ax.axvline(probe_1_x, color="lime", linestyle="--", linewidth=1.1)
      ax.axvline(probe_2_x, color="yellow", linestyle="--", linewidth=1.1)
      it_val = extract_iter(f)
      ax.set_title(
          f"Iteration {it_val} | U_c = {args.uc:.3f} (P1: x={probe_1_x:.0f},"
          f" P2: x={probe_2_x:.0f})"
      )
      ax.set_ylabel("y [LU]")
      plt.colorbar(im, ax=ax, label="Phase Field $\\phi$ [-]")

    axes[-1].set_xlabel("x [LU]")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, "diagnostics.png"), dpi=180)
    plt.close()

    if frame_data:
      iters = [fd[0] for fd in frame_data]
      lb_series = [fd[1] for fd in frame_data]
      plt.figure(figsize=(10, 4.0))
      plt.plot(
          iters, lb_series, "-o", markersize=3, linewidth=1.5,
          color="#1f77b4", label="Attached jet length $L_b$ [LU]"
      )
      for i, (ev_it, _) in enumerate(breakup_events):
        lbl = "Pinch-off event" if i == 0 else None
        plt.axvline(ev_it, color="#d62728", linestyle=":", linewidth=1.2, alpha=0.8, label=lbl)
      plt.xlabel("Iteration [LU]")
      plt.ylabel("Jet Breakup Length $L_b$ [LU]")
      plt.title(f"Periodic Jet Formation and Breakup Cycle ($U_c = {args.uc:.3f}$)")
      plt.grid(True, alpha=0.3)
      plt.legend()
      plt.tight_layout()
      plt.savefig(os.path.join(args.plot_dir, "breakup_telemetry.png"), dpi=180)
      plt.close()


if __name__ == "__main__":
  main()