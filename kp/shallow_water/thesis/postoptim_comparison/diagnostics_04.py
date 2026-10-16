# -*- coding: utf-8 -*-
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
out_prefix = f"kp/shallow_water/thesis/wyniki_analizy_comparison/{case_name}_2DFFT"
os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

y_center = 154
h_base = 9.55
c_wave = np.sqrt(0.00209424 * h_base)

# Bloki przestrzenne
x_up_min, x_up_max = 1500, 3500      
x_down_min, x_down_max = 4500, 5500  

files = sorted(glob.glob(os.path.join(input_dir, "*.vti")))
if not files:
    files = sorted(glob.glob(os.path.join(input_dir, "*.pvti")))

print(f"Ekstrakcja pelnej macierzy przestrzenno-czasowej z {len(files)} plikow...")

data_up_2d, data_down_2d, iterations = [], [], []
x_up_coords, x_down_coords = None, None

for f in files:
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

print("Obliczanie szerokopasmowego widma 2D FFT...")

fft_up = fftshift(fft2(matrix_up)) / (Nt_up * Nx_up)
fft_down = fftshift(fft2(matrix_down)) / (Nt_down * Nx_down)

freqs_t = fftshift(fftfreq(Nt_up, d=dt))
freqs_x_up = fftshift(fftfreq(Nx_up, d=dx))
freqs_x_down = fftshift(fftfreq(Nx_down, d=dx))

mask_f_positive = freqs_t > 0
valid_freqs = freqs_t[mask_f_positive]
lambda_waves = c_wave / valid_freqs

cr_spectrum = np.zeros_like(valid_freqs)
ct_spectrum = np.zeros_like(valid_freqs)
amp_inc_array = np.zeros_like(valid_freqs)
amp_ref_array = np.zeros_like(valid_freqs)
amp_trans_array = np.zeros_like(valid_freqs)

idx_center_k_up = np.argmin(np.abs(freqs_x_up))
idx_center_k_down = np.argmin(np.abs(freqs_x_down))

for idx, f_val in enumerate(valid_freqs):
    global_idx_f = np.where(freqs_t == f_val)[0][0]
    
    k_slice_up = np.abs(fft_up[global_idx_f, :])
    k_slice_down = np.abs(fft_down[global_idx_f, :])
    
    idx_k_inc = np.argmax(k_slice_up)
    dist_from_center = idx_k_inc - idx_center_k_up
    idx_k_ref = idx_center_k_up - dist_from_center
    
    amp_inc = k_slice_up[idx_k_inc]
    amp_ref = k_slice_up[idx_k_ref]
    
    idx_k_trans = np.argmax(k_slice_down) 
    amp_trans = k_slice_down[idx_k_trans]
    
    amp_inc_array[idx] = amp_inc
    amp_ref_array[idx] = amp_ref
    amp_trans_array[idx] = amp_trans
    
    if amp_inc > 1e-8: 
        cr_spectrum[idx] = amp_ref / amp_inc
        ct_spectrum[idx] = amp_trans / amp_inc

band = (lambda_waves >= 200) & (lambda_waves <= 400)

# PRAWIDLOWA METODA: Wazona energetycznie srednia szerokopasmowa
total_inc_energy = np.sum(amp_inc_array[band]**2)
total_ref_energy = np.sum(amp_ref_array[band]**2)
total_trans_energy = np.sum(amp_trans_array[band]**2)

mean_Cr = np.sqrt(total_ref_energy / total_inc_energy) * 100
mean_Ct = np.sqrt(total_trans_energy / total_inc_energy) * 100

report_text = (
    "RAPORT Z SZEROKOPASMOWEJ ANALIZY 2D FFT (PACZKA FALOWA)\n"
    "=======================================================\n"
    f"Srednie Odbicie Amplitudowe (C_R) w pasmie 200-400 LU: {mean_Cr:.2f}%\n"
    f"Srednia Transmisja Amplitudowa (C_T) w pasmie 200-400 LU: {mean_Ct:.2f}%\n"
)

with open(f"{out_prefix}_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print(report_text)

df_out = pd.DataFrame({
    'Dlugosc_Fali': lambda_waves[band],
    'Cr_Amp': cr_spectrum[band],
    'Ct_Amp': ct_spectrum[band]
})
df_out.to_csv(f"{out_prefix}_widmo.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(lambda_waves[band], cr_spectrum[band]*100, 'r--', label='Odbicie C_R', linewidth=2)
plt.plot(lambda_waves[band], ct_spectrum[band]*100, 'g-', label='Transmisja C_T', linewidth=2)
plt.axvspan(200, 400, color='gray', alpha=0.15, label='Glowne pasmo paczki')
plt.title("Charakterystyka Widmowa z Dekonstrukcji 2D FFT")
plt.xlabel("Dlugosc fali [LU]")
plt.ylabel("Wspolczynnik [%]")
plt.xlim(200, 400)
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig(f"{out_prefix}_widmo.png", dpi=300)
plt.close()
