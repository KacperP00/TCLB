# -*- coding: utf-8 -*-
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
lambda_target = 486.0
y_center = 231
h_base = 9.55

# Paski pomiarowe
x_up_min, x_up_max = 2250,  5250     
x_down_min, x_down_max = 6750, 9750  

files = sorted(glob.glob(os.path.join(input_dir, "*.vti")))
if not files:
    files = sorted(glob.glob(os.path.join(input_dir, "*.pvti")))

files = [f for f in files if int(f.split('_')[-1].split('.')[0]) >= 75000]
if len(files) < 10:
    print("Zbyt malo klatek do analizy 2D FFT.")
    exit()

print(f"Ekstrakcja macierzy Przestrzen-Czas z {len(files)} plikow VTK...")

data_up_2d, data_down_2d, iterations = [], [], []
x_up_coords, x_down_coords = None, None

for i, f in enumerate(files):
    iter_num = int(f.split('_')[-1].split('.')[0])
    iterations.append(iter_num)
    
    mesh = pv.read(f)
    data_type = "cell_data" if "Rho" in mesh.cell_data else "point_data"
    
    if data_type == "cell_data":
        points = mesh.cell_centers().points
        rho = mesh.cell_data["Rho"]
    else:
        points = mesh.points
        rho = mesh.point_data["Rho"]
    
    mask_y = np.isclose(points[:, 1], y_center, atol=1.0)
    x_c = points[mask_y, 0]
    rho_c = rho[mask_y] - h_base
    
    sort_idx = np.argsort(x_c)
    x_c = x_c[sort_idx]
    rho_c = rho_c[sort_idx]
    
    mask_up = (x_c >= x_up_min) & (x_c <= x_up_max)
    mask_down = (x_c >= x_down_min) & (x_c <= x_down_max)
    
    if x_up_coords is None:
        x_up_coords = x_c[mask_up]
        x_down_coords = x_c[mask_down]
        
    data_up_2d.append(rho_c[mask_up])
    data_down_2d.append(rho_c[mask_down])

matrix_up = np.array(data_up_2d)
matrix_down = np.array(data_down_2d)

t = np.array(iterations)
dt = np.mean(np.diff(t))
dx = np.mean(np.diff(x_up_coords))

# Usuniecie DC i aplikacja okna Hanna (zapobieganie wyciekom widma)
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

k_spectrum_up = np.abs(fft_up[idx_f, :])

# EKSTRAKCJA SYMETRYCZNA (Ignoruje stacjonarny szum)
idx_k_zero_up = np.argmin(np.abs(freqs_x_up))
idx_k_inc = np.argmax(k_spectrum_up) 
dist_from_center = idx_k_inc - idx_k_zero_up
idx_k_ref = idx_k_zero_up - dist_from_center 

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
plt.axhline(f_target, color='white', linestyle='--', label=f'Wymuszenie {f_target:.5f} Hz')
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
    f"Wspolczynnik Odbicia Rzeczywistego (C_R): {Cr * 100:.2f}%\n"
    f"Wspolczynnik Transmisji (C_T): {Ct * 100:.2f}%\n"
    f"Dyssypacja (Tlumienie + Absorpcja): {Dissipation * 100:.2f}%\n"
)

with open(f"{out_prefix}.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
