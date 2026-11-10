import os
import sys
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# Argumenty: 1: skala (np. 1.0), 2: tryb ("packet" lub "single"), 3: plik wyjsciowy (.mp4 lub .gif)
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
mode = sys.argv[2] if len(sys.argv) > 2 else "packet"
out_file = sys.argv[3] if len(sys.argv) > 3 else f"anim_sync_{mode}_{scale:.1f}x.mp4"

H_BASE = 9.55
VMIN, VMAX = -0.02, 0.02
X_MIN, X_MAX = int(round(3200 * scale)), int(round(5200 * scale))
T_START, T_END = int(round(26000 * scale)), int(round(66000 * scale))

# Konfiguracja 4 katalogow i opcjonalnych plikow STL do nalozenia maski scian
CASES = [
    ("1. Topologia Pierwotna", f"output/validation_scale_{scale:.1f}x/topology_{mode}_{scale:.1f}x", f"kp/shallow_water/thesis/topology_{scale:.1f}x.stl"),
    ("2. Topologia Oczyszczona (area >= 50)", f"output/validation_clean_scale_{scale:.1f}x/topology_{mode}_{scale:.1f}x", f"kp/shallow_water/thesis/topology_clean_{scale:.1f}x.stl"),
    ("3. Model analityczny Euve et al.", f"output/validation_scale_{scale:.1f}x/euve_{mode}_{scale:.1f}x", None),
    ("4. Puste Wneki (brak topologii)", f"output/validation_scale_{scale:.1f}x/empty_cav_{mode}_{scale:.1f}x", None),
]

def iter_of(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])

def read_2d_crop(vti_path):
    # Odczyt wycinka 2D pola odchylenia eta = Rho - H_BASE w oknie [X_MIN, X_MAX]
    mesh = pv.read(vti_path)
    nx, ny = mesh.dimensions[0] - 1, mesh.dimensions[1] - 1
    arr = np.asarray(mesh.cell_data["Rho"]).reshape(-1, ny, nx)[0].astype(np.float64)
    return arr[:, X_MIN:X_MAX] - H_BASE

def build_mask(case_label, sample_vti, stl_path):
    # Utworzenie maski scian kanalu, wnek oraz geometrii STL w oknie [X_MIN, X_MAX]
    mesh = pv.read(sample_vti)
    nx, ny = mesh.dimensions[0] - 1, mesh.dimensions[1] - 1
    mask = np.zeros((ny, nx), dtype=bool)
    c_bot, c_top = int(round(104 * scale)), int(round(204 * scale))
    x_in, x_out = int(round(4000 * scale)), int(round(4400 * scale))

    # Sciany kanalu poza wnekami
    mask[:c_bot, :x_in] = True; mask[:c_bot, x_out:] = True
    mask[c_top:, :x_in] = True; mask[c_top:, x_out:] = True

    if "Euve" in case_label:
        mask[c_top:, :] = True
        mask[:c_bot, :] = True
        c1_dx, c_nx = int(round(4154 * scale)), int(round(44 * scale))
        c1_ny = c_bot - int(round(54 * scale))
        c2_dx = c1_dx + c_nx + int(round(4 * scale))
        c2_ny = c_bot - int(round(61 * scale))
        mask[c1_ny:c_bot, c1_dx:c1_dx + c_nx] = False
        mask[c2_ny:c_bot, c2_dx:c2_dx + c_nx] = False
    else:
        mask[:int(round(4 * scale)), x_in:x_out] = True
        mask[int(round(304 * scale)):, x_in:x_out] = True
        if stl_path and os.path.exists(stl_path):
            stl = pv.read(stl_path)
            pts = stl.points[np.isclose(stl.points[:, 2], 1.0)]
            ix = np.clip(np.floor(pts[:, 0]).astype(int), 0, nx - 1)
            iy = np.clip(np.floor(pts[:, 1]).astype(int), 0, ny - 1)
            mask[iy, ix] = True

    return mask[:, X_MIN:X_MAX]

if __name__ == "__main__":
    case_files, masks = [], []
    for label, folder, stl_p in CASES:
        flist = sorted(glob.glob(os.path.join(folder, "*.vti")), key=iter_of)
        flist = [f for f in flist if T_START <= iter_of(f) <= T_END]
        if not flist:
            sys.exit(f"BLAD: Brak plikow .vti w katalogu {folder} (sprawdz czy rm *.vti ich nie usunal).")
        case_files.append(flist)
        masks.append(build_mask(label, flist[0], stl_p))

    n_frames = min(len(fl) for fl in case_files)
    its = [iter_of(f) for f in case_files[0][:n_frames]]
    ny_crop = masks[0].shape[0]

    fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True, sharey=True)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#1e293b")

    ims = []
    for ax, (label, _, _), msk, flist in zip(axes, CASES, masks, case_files):
        eta0 = np.ma.masked_where(msk, read_2d_crop(flist[0]))
        im = ax.imshow(eta0, origin="lower", extent=[X_MIN, X_MAX, 0, ny_crop],
                       cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="auto")
        ax.axvline(4000 * scale, color="k", linestyle="--", lw=1.0, alpha=0.6)
        ax.axvline(4400 * scale, color="k", linestyle="--", lw=1.0, alpha=0.6)
        ax.set_title(label, fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("Y [LU]")
        ims.append(im)

    axes[-1].set_xlabel(f"Wspolrzedna X (okolice absorbera: wneka + 2*L z obu stron, skala {scale:.1f}x) [LU]")
    suptitle = fig.suptitle(f"Ewolucja pola falowego ({mode}) | Iteracja: {its[0]}", fontsize=12, fontweight="bold")
    fig.colorbar(ims[0], ax=axes.ravel().tolist(), orientation="vertical", fraction=0.018, pad=0.02, label="eta = Rho - 9.55 [LU]")

    def update(idx):
        for im, msk, flist in zip(ims, masks, case_files):
            im.set_data(np.ma.masked_where(msk, read_2d_crop(flist[idx])))
        suptitle.set_text(f"Ewolucja pola falowego ({mode}) | Iteracja: {its[idx]}")
        return ims + [suptitle]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    writer = FFMpegWriter(fps=12, bitrate=3000, codec="libx264") if out_file.endswith(".mp4") else FFMpegWriter(fps=25, bitrate=3000)
    anim.save(out_file, writer=writer, dpi=150)
    plt.close(fig)
    print(f"Zapisano zsynchronizowana animacje: {out_file}")