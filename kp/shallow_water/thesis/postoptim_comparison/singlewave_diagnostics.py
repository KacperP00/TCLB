import pyvista as pv
import sys
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq

# Konfiguracja
input_dir = sys.argv[1] if len(sys.argv) > 1 else "output/"
# Pobranie nazwy przypadku z drugiego argumentu
case_name = sys.argv[2] if len(sys.argv) > 2 else "unknown_case"
out_prefix = f"kp/shallow_water/thesis/wyniki_analizy_comparison/{case_name}_2DFFT"
os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

# Parametry fizyczne
lambda_target = 294.0
y_center = 154
h_base = 9.55

# Paski pomiarowe (zamiast punktów bierzemy bloki przestrzeni)
x_up_min, x_up_max = 1500, 3500      # Strefa badana przed przeszkodą (Odbicie)
x_down_min, x_down_max = 4500, 5500  # Strefa badana za przeszkodą (Transmisja)

files = sorted(glob.glob(os.path.join(input_dir, "*.vti")))
if not files:
    files = sorted(glob.glob(os.path.join(input_dir, "*.pvti")))

# Analiza stanu ustalonego (bierzemy ostatnie klatki)
# Pobierz wyłącznie pliki ze stanu ustalonego
files = [f for f in files if int(f.split('_')[-1].split('.')[0]) >= 50000]
if len(files) < 10:
    print("Zbyt mało klatek do analizy 2D FFT.")
    exit()

print(f"Ekstrakcja macierzy Przestrzeń-Czas z {len(files)} plików VTK...")

data_up_2d = []
data_down_2d = []
iterations = []
x_up_coords = None
x_down_coords = None

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
    
    # Sortowanie wzdłuż kanału
    sort_idx = np.argsort(x_c)
    x_c = x_c[sort_idx]
    rho_c = rho_c[sort_idx]
    
    # Ekstrakcja pasków (Upstream i Downstream)
    mask_up = (x_c >= x_up_min) & (x_c <= x_up_max)
    mask_down = (x_c >= x_down_min) & (x_c <= x_down_max)
    
    if x_up_coords is None:
        x_up_coords = x_c[mask_up]
        x_down_coords = x_c[mask_down]
        
    data_up_2d.append(rho_c[mask_up])
    data_down_2d.append(rho_c[mask_down])

# Budowa macierzy: wiersze = czas (t), kolumny = przestrzeń (x)
matrix_up = np.array(data_up_2d)
matrix_down = np.array(data_down_2d)

t = np.array(iterations)
dt = np.mean(np.diff(t))
dx = np.mean(np.diff(x_up_coords))

# Usunięcie składowej stałej (DC) dla czystego widma
matrix_up -= np.mean(matrix_up)
matrix_down -= np.mean(matrix_down)

print("Kalkulacja 2D FFT (Analiza częstotliwość - liczba falowa)...")

# --- 2D FFT DLA STREFY ODBICIA (UPSTREAM) ---
fft_up = fftshift(fft2(matrix_up))
Nt_up, Nx_up = matrix_up.shape

freqs_t = fftshift(fftfreq(Nt_up, d=dt))
freqs_x = fftshift(fftfreq(Nx_up, d=dx)) # Liczba falowa (k / 2pi)

# Izolacja właściwej częstotliwości uderzenia
c_wave = np.sqrt(0.00209424 * h_base)
f_target = c_wave / lambda_target
idx_f = np.argmin(np.abs(freqs_t - f_target))

# Ekstrakcja przekroju widma przestrzennego (zależności od kierunku płynięcia)
# Dla f > 0: ujemne k_x (freqs_x < 0) to fala w prawo (Inc), dodatnie k_x to fala w lewo (Ref)
# W konwencji numpy.fftfreg zależy to od definicji fazy, szukamy dwóch największych pików.
k_spectrum_up = np.abs(fft_up[idx_f, :])

# Znalezienie fali padającej i odbitej poprzez separację osi k (liczby falowej)
idx_k_zero = np.argmin(np.abs(freqs_x))
peak_inc = np.max(k_spectrum_up[:idx_k_zero])  # Pik fali płynącej do przodu
peak_ref = np.max(k_spectrum_up[idx_k_zero:])  # Pik fali wracającej po odbiciu

Cr = peak_ref / peak_inc

# --- 2D FFT DLA STREFY TRANSMISJI (DOWNSTREAM) ---
fft_down = fftshift(fft2(matrix_down))
Nt_down, Nx_down = matrix_down.shape
freqs_x_down = fftshift(fftfreq(Nx_down, d=dx))

idx_k_zero_down = np.argmin(np.abs(freqs_x_down))
k_spectrum_down = np.abs(fft_down[idx_f, :])
peak_trans = np.max(k_spectrum_down[:idx_k_zero_down]) # Transmisja płynie tylko do przodu

Ct = peak_trans / peak_inc
Dissipation = 1.0 - (Cr**2) - (Ct**2)

# --- GENEROWANIE SPEKTROGRAMU f-k DLA ODBICIA ---
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(fft_up), aspect='auto', extent=[freqs_x[0], freqs_x[-1], freqs_t[-1], freqs_t[0]], cmap='jet')
plt.axhline(f_target, color='white', linestyle='--', label=f'Wymuszenie {f_target:.5f} Hz')
plt.axvline(0, color='white', linestyle='-')
plt.colorbar(label='Amplituda Widmowa')
plt.title("Spektrogram 2D (k-omega) - Strefa przed rezonatorem")
plt.xlabel("Liczba falowa (Kierunek: <0 Wlot, >0 Odbicie)")
plt.ylabel("Częstotliwość f [1/iter]")
plt.ylim(0, f_target * 3) # Przybliżenie widoku do obszaru roboczego
plt.xlim(-0.01, 0.01)
plt.legend()
plt.savefig(f"{out_prefix}_spectrogram.png", dpi=300)
plt.close()

# Wyrzut danych
report_text = (
    "RAPORT Z ANALIZY 2D FFT (KIERUNKOWEJ)\n"
    "=====================================\n"
    f"Współczynnik Odbicia Rzeczywistego (C_R): {Cr * 100:.2f}%\n"
    f"Współczynnik Transmisji (C_T): {Ct * 100:.2f}%\n"
    f"Dyssypacja (Tłumienie + Absorpcja): {Dissipation * 100:.2f}%\n"
)

with open(f"{out_prefix}.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)