# -*- coding: utf-8 -*-
# Single (continuous) wave reflection/transmission analysis (2D FFT, dense 1.5x geometry).
# Changes w.r.t. the previous version are tagged FIX-n (see comments).
import pyvista as pv
import sys
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq
from scipy.signal.windows import hann

# Konfiguracja
input_dir = sys.argv[1] if len(sys.argv) > 1 else "output/"
case_name = sys.argv[2] if len(sys.argv) > 2 else "unknown_case"
out_prefix = f"kp/shallow_water/thesis/wyniki_analizy_comparison_dense1.5/{case_name}_2DFFT"
os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

# Parametry fizyczne
# FIX-5: wavelength comes from the command line (run_comparison.sh passes the same
# value to singlewave_generator.py and to this script); 486 is only a default.
lambda_target = float(sys.argv[3]) if len(sys.argv) > 3 else 486.0
# FIX-1: channel = y in [156, 306); average over its width minus a wall margin
Y_CH_MIN, Y_CH_MAX, Y_MARGIN = 156, 306, 10
h_base = 9.55

# Paski pomiarowe
x_up_min, x_up_max = 2250,  5250
x_down_min, x_down_max = 6750, 9750

# Analysed frames: steady state only (the transmitted front passes x=9750 at ~69000).
# Checked: the wave re-reflected at the WPressure inlet re-enters the upstream strip only
# after ~100800; with the temporal Hann window below this changes Cr by < 0.05%, so the
# frames are NOT cut at the end.
T_START = 75000


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

files = sorted(files, key=iter_of)                                   # FIX-6
files = [f for f in files if iter_of(f) >= T_START]
if len(files) < 10:
    print("Zbyt malo klatek do analizy 2D FFT.")
    exit()

print(f"Ekstrakcja macierzy Przestrzen-Czas z {len(files)} plikow VTK...")

data_up_2d, data_down_2d, iterations = [], [], []
x_up_coords, x_down_coords = None, None

for i, f in enumerate(files):
    iterations.append(iter_of(f))
    x_c, rho_c = read_profile(f)

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
if not np.allclose(dts, dts[0]):                                     # FIX-6
    sys.exit(f"Non-uniform VTK frame spacing: {np.unique(dts)}")
dt = dts[0]
dx = np.diff(x_up_coords)[0]
print(f"Nt={len(t)}, dt={dt:g}, Nx_up={len(x_up_coords)}, dx={dx:g}")
if abs(dx - 1.0) > 1e-9:
    print("WARNING: dx != 1 LU - wavelengths are only valid if the lattice unit is 1")

# Usuniecie DC i aplikacja okna Hanna (zapobieganie wyciekom widma)
# (temporal Hann is fine here: all components are stationary over the analysed frames)
matrix_up -= np.mean(matrix_up)
matrix_down -= np.mean(matrix_down)

Nt_up, Nx_up = matrix_up.shape
Nt_down, Nx_down = matrix_down.shape

window_up = hann(Nt_up)[:, None] * hann(Nx_up)[None, :]
window_down = hann(Nt_down)[:, None] * hann(Nx_down)[None, :]

matrix_up = matrix_up * window_up
matrix_down = matrix_down * window_down

print("Kalkulacja 2D FFT (Analiza czestotliwosc - liczba falowa)...")

# --- 2D FFT DLA STREFY ODBICIA (UPSTREAM) ---
fft_up = fftshift(fft2(matrix_up)) / (Nt_up * Nx_up)

freqs_t = fftshift(fftfreq(Nt_up, d=dt))
freqs_x_up = fftshift(fftfreq(Nx_up, d=dx))

c_wave = np.sqrt(0.00209424 * h_base)
f_target = c_wave / lambda_target
idx_f = np.argmin(np.abs(freqs_t - f_target))
print(f"lambda={lambda_target:g} LU, f_target={f_target:.4e}, nearest bin offset = "
      f"{(freqs_t[idx_f]-f_target)/(freqs_t[1]-freqs_t[0]):+.2f} bin")

k_spectrum_up = np.abs(fft_up[idx_f, :])

# EKSTRAKCJA SYMETRYCZNA (Ignoruje stacjonarny szum)
idx_k_zero_up = np.argmin(np.abs(freqs_x_up))
# FIX-3: incident (right-going) peak searched only in the k<0 half
idx_k_inc = np.argmax(k_spectrum_up[:idx_k_zero_up])
idx_k_ref = min(2 * idx_k_zero_up - idx_k_inc, Nx_up - 1)

peak_inc = k_spectrum_up[idx_k_inc]
peak_ref = k_spectrum_up[idx_k_ref]

Cr = peak_ref / peak_inc

# --- 2D FFT DLA STREFY TRANSMISJI (DOWNSTREAM) ---
fft_down = fftshift(fft2(matrix_down)) / (Nt_down * Nx_down)
freqs_x_down = fftshift(fftfreq(Nx_down, d=dx))

idx_k_zero_down = np.argmin(np.abs(freqs_x_down))
k_spectrum_down = np.abs(fft_down[idx_f, :])
peak_trans = np.max(k_spectrum_down[:idx_k_zero_down])

Ct = peak_trans / peak_inc
Dissipation = 1.0 - (Cr**2) - (Ct**2)

# --- GENEROWANIE SPEKTROGRAMU f-k DLA ODBICIA ---
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(fft_up), aspect='auto', extent=[freqs_x_up[0], freqs_x_up[-1], freqs_t[-1], freqs_t[0]], cmap='jet')
plt.axhline(f_target, color='white', linestyle='--', label=f'Wymuszenie {f_target:.5f} [1/iter]')
plt.axvline(0, color='white', linestyle='-')
plt.colorbar(label='Amplituda Widmowa')
plt.title("Spektrogram 2D (k-omega) - Strefa przed rezonatorem")
plt.xlabel("Liczba falowa (Kierunek: <0 Wlot, >0 Odbicie)")
plt.ylabel("Czestotliwosc f [1/iter]")
plt.ylim(0, f_target * 3)
plt.xlim(-0.01, 0.01)
plt.legend()
plt.savefig(f"{out_prefix}_spectrogram.png", dpi=300)
plt.close()

# Wyrzut danych
report_text = (
    "RAPORT Z ANALIZY 2D FFT (KIERUNKOWEJ)\n"
    "=====================================\n"
    f"Dlugosc fali: {lambda_target:g} LU, klatki {iterations[0]}-{iterations[-1]} ({len(iterations)} szt.)\n"
    f"Wspolczynnik Odbicia Rzeczywistego (C_R): {Cr * 100:.2f}%\n"
    f"Wspolczynnik Transmisji (C_T): {Ct * 100:.2f}%\n"
    f"Dyssypacja (Tlumienie + Absorpcja): {Dissipation * 100:.2f}%\n"
)

with open(f"{out_prefix}.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
