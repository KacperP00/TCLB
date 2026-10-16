import pyvista as pv
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq
import pandas as pd

# Konfiguracja - Zmien na odpowiedni folder z PACZKa FALOWa
input_dir = sys.argv[1] if len(sys.argv) > 1 else "output/" 
case_name = sys.argv[2] if len(sys.argv) > 2 else "unknown_case"
out_prefix = f"kp/shallow_water/thesis/wyniki_analizy_comparison/{case_name}_2DFFT"
os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

y_center = 154
h_base = 9.55
c_wave = np.sqrt(0.00209424 * h_base)

# Bloki przestrzenne (Paski analizy)
x_up_min, x_up_max = 1500, 3500      
x_down_min, x_down_max = 4500, 5500  

# Dla paczki falowej wczytujemy CAlY proces symulacji (nie tylko stan ustalony)
files = sorted(glob.glob(os.path.join(input_dir, "*.vti")))
if not files:
    files = sorted(glob.glob(os.path.join(input_dir, "*.pvti")))

print(f"Ekstrakcja penej macierzy przestrzenno-czasowej z {len(files)} plikw...")

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

# Usuniecie skladowej stalej (wyplaszczenie poziomu morza)
matrix_up -= np.mean(matrix_up)
matrix_down -= np.mean(matrix_down)

print("Obliczanie szerokopasmowego widma 2D FFT...")
Nt, Nx_up = matrix_up.shape
_, Nx_down = matrix_down.shape

# Normalizacja amplitud przez rozmiar dziedziny (Nt * Nx)
fft_up = fftshift(fft2(matrix_up)) / (Nt * Nx_up)
fft_down = fftshift(fft2(matrix_down)) / (Nt * Nx_down)
freqs_t = fftshift(fftfreq(Nt, d=dt))
freqs_x_up = fftshift(fftfreq(Nx_up, d=dx))
freqs_x_down = fftshift(fftfreq(Nx_down, d=dx))

# Szukamy tylko dodatnich czestotliwosci
mask_f_positive = freqs_t > 0
valid_freqs = freqs_t[mask_f_positive]
lambda_waves = c_wave / valid_freqs

# Inicjalizacja wynikow
cr_spectrum = np.zeros_like(valid_freqs)
ct_spectrum = np.zeros_like(valid_freqs)
idx_center_k_up = np.argmin(np.abs(freqs_x_up))
idx_center_k_down = np.argmin(np.abs(freqs_x_down))

# Dekonstrukcja kierunkowa dla kazdej czestotliwosci w paczce falowej
for idx, f_val in enumerate(valid_freqs):
    global_idx_f = np.where(freqs_t == f_val)[0][0]
    
    # Przekroj poziomy widma k (Liczba falowa) dla danej czestotliwosci f
    k_slice_up = np.abs(fft_up[global_idx_f, :])
    k_slice_down = np.abs(fft_down[global_idx_f, :])
    
    # W paczce falowej dominujacym sygnalem jest zawsze fala padajaca (max pik)
    idx_k_inc = np.argmax(k_slice_up)
    
    # Fala odbita znajduje sie matematycznie po przeciwnej stronie osi k (symetria kierunku)
    dist_from_center = idx_k_inc - idx_center_k_up
    idx_k_ref = idx_center_k_up - dist_from_center
    
    amp_inc = k_slice_up[idx_k_inc]
    amp_ref = k_slice_up[idx_k_ref]
    
    # Podobnie dla transmisji (szukamy fali biegnacej w tym samym kierunku co padajaca)
    idx_k_trans = np.argmax(k_slice_down) 
    amp_trans = k_slice_down[idx_k_trans]
    
    if amp_inc > 1e-8: # Zabezpieczenie przed dzieleniem przez zero dla szumu
        cr_spectrum[idx] = amp_ref / amp_inc
        ct_spectrum[idx] = amp_trans / amp_inc

# Ograniczenie do uzytecznego pasma energetycznego paczki (200 - 400 LU)
band = (lambda_waves >= 200) & (lambda_waves <= 400)
mean_Cr = np.mean(cr_spectrum[band]) * 100
mean_Ct = np.mean(ct_spectrum[band]) * 100

report_text = (
    "RAPORT Z SZEROKOPASMOWEJ ANALIZY 2D FFT (PACZKA FALOWA)\n"
    "=======================================================\n"
    f"srednie Odbicie Amplitudowe (C_R) w pasmie 200-400 LU: {mean_Cr:.2f}%\n"
    f"srednia Transmisja Amplitudowa (C_T) w pasmie 200-400 LU: {mean_Ct:.2f}%\n"
)

with open(f"{out_prefix}_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print(report_text)

# Zapis widma
df_out = pd.DataFrame({
    'Dlugosc_Fali': lambda_waves[band],
    'Cr_Amp': cr_spectrum[band],
    'Ct_Amp': ct_spectrum[band]
})
df_out.to_csv(f"{out_prefix}_widmo.csv", index=False)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))
plt.plot(lambda_waves[band], cr_spectrum[band]*100, 'r--', label='Odbicie C_R', linewidth=2)
plt.plot(lambda_waves[band], ct_spectrum[band]*100, 'g-', label='Transmisja C_T', linewidth=2)
plt.axvspan(200, 400, color='gray', alpha=0.15, label='Glowne pasmo paczki')
plt.title("Charakterystyka Widmowa z Dekonstrukcji 2D FFT")
plt.xlabel("Dlugosc fali ? [LU]")
plt.ylabel("Wspolczynnik [%]")
plt.xlim(200, 400)
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig(f"{out_prefix}_widmo.png", dpi=300)
plt.close()
