# -*- coding: utf-8 -*-
# Wave-packet reflection/transmission analysis (2D FFT).
# Usage: diagnostics.py <vti_dir> <case_name> [dense|old]
# Changes w.r.t. the previous version are tagged FIX-n (see comments).
import pyvista as pv
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq
from scipy.signal.windows import hann
import pandas as pd

# Konfiguracja
input_dir = sys.argv[1] if len(sys.argv) > 1 else "output/" 
case_name = sys.argv[2] if len(sys.argv) > 2 else "unknown_case"

# FIX-7: geometry preset (3rd argument). The strips, the channel position and the band
# MUST match the grid that produced the VTK files:
#   dense : 12624 x 462 grid, channel y in [156,306), packet band 300-600 LU
#   old   :  8416 x 308 grid, channel y in [104,204), packet band 200-400 LU
GEOMETRIES = {
    "dense": dict(y=(156, 306), up=(2250, 5250), down=(6750, 9750), band=(300, 600),
                  out="kp/shallow_water/thesis/wyniki_analizy_comparison_dense1.5"),
    "old":   dict(y=(104, 204), up=(1500, 3500), down=(4500, 5500), band=(200, 400),
                  out="kp/shallow_water/thesis/wyniki_analizy_comparison"),
}
geom_name = sys.argv[3] if len(sys.argv) > 3 else "dense"
if geom_name not in GEOMETRIES:
    sys.exit(f"Unknown geometry '{geom_name}', use one of {list(GEOMETRIES)}")
G = GEOMETRIES[geom_name]
print(f"Geometry preset: {geom_name}")

out_prefix = f"{G['out']}/{case_name}_2DFFT"
os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

# FIX-1: the water channel (walls on both sides). The profile is averaged over the
# channel width (minus a margin at the walls) instead of picking "rows within
# atol=1.0 of the centre line" (which matched 2-3 rows).
Y_CH_MIN, Y_CH_MAX = G["y"]
Y_MARGIN = 10
h_base = 9.55
c_wave = np.sqrt(0.00209424 * h_base)

# Bloki przestrzenne
x_up_min, x_up_max = G["up"]
x_down_min, x_down_max = G["down"]
BAND_MIN, BAND_MAX = G["band"]

# FIX-4: a frequency bin is trusted only if the incident amplitude there is at
# least this fraction of its maximum over the band (the Gaussian packet has only
# ~0.4% of its peak amplitude at the 300 / 600 LU band edges -> pure noise there)
VALID_FRAC = 0.10


def iter_of(path):
    return int(os.path.basename(path).split('_')[-1].split('.')[0])


def read_profile(path):
    """One VTK frame -> (x, rho averaged over channel width - h_base)."""
    mesh = pv.read(path)
    if hasattr(mesh, "dimensions"):                      # ImageData (.vti / merged .pvti)
        if "Rho" in mesh.cell_data:
            arr, off = np.asarray(mesh.cell_data["Rho"]), 0.5
            nx, ny = mesh.dimensions[0] - 1, mesh.dimensions[1] - 1
        else:
            arr, off = np.asarray(mesh.point_data["Rho"]), 0.0
            nx, ny = mesh.dimensions[0], mesh.dimensions[1]
        if arr.size % (nx * ny) != 0:
            sys.exit(f"Unexpected Rho size {arr.size} for grid {nx}x{ny}")
        field = arr.reshape(-1, ny, nx)[0].astype(np.float64)   # VTK: x runs fastest
        x = mesh.origin[0] + (np.arange(nx) + off) * mesh.spacing[0]
        y = mesh.origin[1] + (np.arange(ny) + off) * mesh.spacing[1]
        rows = (y >= Y_CH_MIN + Y_MARGIN) & (y <= Y_CH_MAX - Y_MARGIN)
        return x, field[rows, :].mean(axis=0) - h_base
    # generic fallback (any dataset type): group by x and average over y
    if "Rho" in mesh.cell_data:
        pts, rho = mesh.cell_centers().points, np.asarray(mesh.cell_data["Rho"])
    else:
        pts, rho = mesh.points, np.asarray(mesh.point_data["Rho"])
    m = (pts[:, 1] >= Y_CH_MIN + Y_MARGIN) & (pts[:, 1] <= Y_CH_MAX - Y_MARGIN)
    x, inv = np.unique(pts[m, 0], return_inverse=True)
    return x, np.bincount(inv, weights=rho[m]) / np.bincount(inv) - h_base


files = glob.glob(os.path.join(input_dir, "*.vti"))
if not files:
    files = glob.glob(os.path.join(input_dir, "*.pvti"))
files = sorted(files, key=iter_of)          # FIX-6: sort by iteration number, not by string

print(f"Ekstrakcja pelnej macierzy przestrzenno-czasowej z {len(files)} plikow...")

data_up_2d, data_down_2d, iterations = [], [], []
x_up_coords, x_down_coords = None, None

for f in files:
    iterations.append(iter_of(f))
    x_c, rho_c = read_profile(f)

    if x_up_coords is None and (x_c.min() > x_up_min or x_c.max() < x_down_max):
        sys.exit(f"Measurement strips ({x_up_min}-{x_down_max}) do not fit in the data "
                 f"(x = {x_c.min():.0f}..{x_c.max():.0f}). Wrong geometry preset? "
                 f"Current: '{geom_name}' (use: diagnostics.py <dir> <case> dense|old)")

    mask_up = (x_c >= x_up_min) & (x_c <= x_up_max)
    mask_down = (x_c >= x_down_min) & (x_c <= x_down_max)

    if x_up_coords is None:
        x_up_coords = x_c[mask_up]
        x_down_coords = x_c[mask_down]

    data_up_2d.append(rho_c[mask_up])
    data_down_2d.append(rho_c[mask_down])

matrix_up = np.array(data_up_2d)
matrix_down = np.array(data_down_2d)

t = np.array(iterations, dtype=float)
dts = np.diff(t)
if not np.allclose(dts, dts[0]):                      # FIX-6
    sys.exit(f"Non-uniform VTK frame spacing: {np.unique(dts)}")
dt = dts[0]
dx = np.diff(x_up_coords)[0]
print(f"Nt={len(t)}, dt={dt:g}, Nx_up={len(x_up_coords)}, Nx_down={len(x_down_coords)}, dx={dx:g}")
if abs(dx - 1.0) > 1e-9:
    print("WARNING: dx != 1 LU - wavelengths below are only valid if the lattice unit is 1")

# Usuniecie DC
matrix_up -= np.mean(matrix_up)
matrix_down -= np.mean(matrix_down)

Nt_up, Nx_up = matrix_up.shape
Nt_down, Nx_down = matrix_down.shape

# FIX-2: spatial Hann window only. A temporal Hann window multiplies the incident,
# reflected and transmitted packets by DIFFERENT weights (they cross the measurement
# strips at different times: ~t=49000, ~81000 and ~81000 of 106500), which biases
# Cr and Ct (typically low). The packet already decays to ~0 at both ends of the
# record, so no temporal window is needed.
window_up = hann(Nx_up)[None, :]
window_down = hann(Nx_down)[None, :]

matrix_up = matrix_up * window_up
matrix_down = matrix_down * window_down

print("Obliczanie szerokopasmowego widma 2D FFT...")

fft_up = fftshift(fft2(matrix_up)) / (Nt_up * Nx_up)
fft_down = fftshift(fft2(matrix_down)) / (Nt_down * Nx_down)

freqs_t = fftshift(fftfreq(Nt_up, d=dt))
freqs_x_up = fftshift(fftfreq(Nx_up, d=dx))
freqs_x_down = fftshift(fftfreq(Nx_down, d=dx))

idx_pos = np.where(freqs_t > 0)[0]
valid_freqs = freqs_t[idx_pos]
lambda_waves = c_wave / valid_freqs

cr_spectrum = np.zeros_like(valid_freqs)
ct_spectrum = np.zeros_like(valid_freqs)
amp_inc_array = np.zeros_like(valid_freqs)
amp_ref_array = np.zeros_like(valid_freqs)
amp_trans_array = np.zeros_like(valid_freqs)

idx_center_k_up = np.argmin(np.abs(freqs_x_up))
idx_center_k_down = np.argmin(np.abs(freqs_x_down))

for n, idx_f in enumerate(idx_pos):
    k_slice_up = np.abs(fft_up[idx_f, :])
    k_slice_down = np.abs(fft_down[idx_f, :])

    # FIX-3: at f>0 a right-going wave sits at k<0. Search the incident peak only
    # in the k<0 half; the reflected wave is the mirror bin (k>0). The transmitted
    # peak is searched only in the k<0 half of the downstream strip as well.
    idx_k_inc = np.argmax(k_slice_up[:idx_center_k_up])
    idx_k_ref = min(2 * idx_center_k_up - idx_k_inc, Nx_up - 1)

    amp_inc = k_slice_up[idx_k_inc]
    amp_ref = k_slice_up[idx_k_ref]
    amp_trans = np.max(k_slice_down[:idx_center_k_down])

    amp_inc_array[n] = amp_inc
    amp_ref_array[n] = amp_ref
    amp_trans_array[n] = amp_trans

    if amp_inc > 1e-8:
        cr_spectrum[n] = amp_ref / amp_inc
        ct_spectrum[n] = amp_trans / amp_inc

band = (lambda_waves >= BAND_MIN) & (lambda_waves <= BAND_MAX)

# FIX-7: the incident spectrum must peak inside the analysed band, otherwise the band
# (preset) does not belong to this packet and every ratio below would be noise
near = (lambda_waves >= 0.5 * BAND_MIN) & (lambda_waves <= 2.0 * BAND_MAX)
lam_peak = lambda_waves[near][np.argmax(amp_inc_array[near])]
print(f"Incident spectrum peaks at lambda = {lam_peak:.0f} LU (band {BAND_MIN}-{BAND_MAX})")
if not (BAND_MIN <= lam_peak <= BAND_MAX):
    sys.exit(f"Incident spectrum peaks at {lam_peak:.0f} LU, outside the band {BAND_MIN}-{BAND_MAX} "
             f"of preset '{geom_name}'. Wrong geometry preset or wrong packet?")

# FIX-4: trusted bins = in band AND incident amplitude >= VALID_FRAC * band maximum
valid = band & (amp_inc_array >= VALID_FRAC * amp_inc_array[band].max())
if not valid.any():
    sys.exit("No valid frequency bins - incident wave not found in the upstream strip")

# Wazona energetycznie srednia szerokopasmowa (tylko biny wiarygodne)
total_inc_energy = np.sum(amp_inc_array[valid]**2)
total_ref_energy = np.sum(amp_ref_array[valid]**2)
total_trans_energy = np.sum(amp_trans_array[valid]**2)

mean_Cr = np.sqrt(total_ref_energy / total_inc_energy) * 100
mean_Ct = np.sqrt(total_trans_energy / total_inc_energy) * 100

lam_lo, lam_hi = lambda_waves[valid].min(), lambda_waves[valid].max()
report_text = (
    "RAPORT Z SZEROKOPASMOWEJ ANALIZY 2D FFT (PACZKA FALOWA)\n"
    "=======================================================\n"
    f"Zakres wiarygodny (amp. padajaca >= {VALID_FRAC*100:.0f}% maksimum): {lam_lo:.0f}-{lam_hi:.0f} LU\n"
    f"Srednie Odbicie Amplitudowe (C_R) w tym zakresie: {mean_Cr:.2f}%\n"
    f"Srednia Transmisja Amplitudowa (C_T) w tym zakresie: {mean_Ct:.2f}%\n"
)

with open(f"{out_prefix}_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print(report_text)

# FIX-4: untrusted bins are written as NaN so that downstream scripts cannot use them
df_out = pd.DataFrame({
    'Dlugosc_Fali': lambda_waves[band],
    'Cr_Amp': np.where(valid, cr_spectrum, np.nan)[band],
    'Ct_Amp': np.where(valid, ct_spectrum, np.nan)[band],
    'Amp_Inc': amp_inc_array[band],
})
df_out.to_csv(f"{out_prefix}_widmo.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(lambda_waves[band], np.where(valid, cr_spectrum, np.nan)[band]*100, 'r--', label='Odbicie C_R', linewidth=2)
plt.plot(lambda_waves[band], np.where(valid, ct_spectrum, np.nan)[band]*100, 'g-', label='Transmisja C_T', linewidth=2)
plt.axvspan(lam_lo, lam_hi, color='gray', alpha=0.15, label='Zakres wiarygodny')
plt.title("Charakterystyka Widmowa z Dekonstrukcji 2D FFT")
plt.xlabel("Dlugosc fali [LU]")
plt.ylabel("Wspolczynnik [%]")
plt.xlim(BAND_MIN, BAND_MAX)
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig(f"{out_prefix}_widmo.png", dpi=300)
plt.close()
