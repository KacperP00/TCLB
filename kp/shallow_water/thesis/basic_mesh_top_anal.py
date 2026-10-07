import pyvista as pv
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# Konfiguracja środowiska i siatki
output_dir = "top_valid_basic_output"
y_center = 154
x_probe_up = 2500      # Sonda przed przeszkodą
x_probe_down = 6000    # Sonda za przeszkodą
h_base = 9.55

files = sorted(glob.glob(os.path.join(output_dir, "*.vti")))
if not files:
    files = sorted(glob.glob(os.path.join(output_dir, "*.pvti")))

# Inicjalizacja tablic dla przebiegów czasowych
iterations = []
eta_up = []
eta_down = []

print("Ekstrakcja przebiegów czasowych z plików VTK...")
for f in files:
    # Ekstrakcja numeru iteracji z nazwy pliku
    iter_num = int(f.split('_')[-1].split('.')[0])
    iterations.append(iter_num)
    
    mesh = pv.read(f)
    data_type = "cell_data" if "Rho" in mesh.cell_data else "point_data"
    
    if data_type == "cell_data":
        x_coords = mesh.cell_centers().points[:, 0]
        y_coords = mesh.cell_centers().points[:, 1]
        rho = mesh.cell_data["Rho"]
    else:
        x_coords = mesh.points[:, 0]
        y_coords = mesh.points[:, 1]
        rho = mesh.point_data["Rho"]
    
    # Lokalizacja indeksów siatki najbliższych sondom
    dist_up = np.sqrt((x_coords - x_probe_up)**2 + (y_coords - y_center)**2)
    dist_down = np.sqrt((x_coords - x_probe_down)**2 + (y_coords - y_center)**2)
    
    idx_up = np.argmin(dist_up)
    idx_down = np.argmin(dist_down)
    
    eta_up.append(rho[idx_up] - h_base)
    eta_down.append(rho[idx_down] - h_base)

t = np.array(iterations)
eta_up = np.array(eta_up)
eta_down = np.array(eta_down)
dt = np.mean(np.diff(t))

# Rozdzielenie sygnału przed przeszkodą (okno czasowe)
# Szacowany punkt podziału między falą padającą a odbitą
split_idx = len(t) // 2  

eta_inc = np.copy(eta_up)
eta_inc[split_idx:] = 0.0  # Wygaszenie fali odbitej

eta_ref = np.copy(eta_up)
eta_ref[:split_idx] = 0.0  # Wygaszenie fali padającej

# Generowanie wykresu kontrolnego w domenie czasu
plt.figure(figsize=(12, 6))
plt.plot(t, eta_inc, label='Fala padająca (X=2500)', color='blue', alpha=0.7)
plt.plot(t, eta_ref, label='Fala odbita (X=2500)', color='red', alpha=0.7)
plt.plot(t, eta_down, label='Transmisja (X=6000)', color='green', alpha=0.7)
plt.title("Przebiegi czasowe h(t) wyekstrahowane z VTK")
plt.xlabel("Iteracja")
plt.ylabel("Wychylenie [LU]")
plt.legend()
plt.grid(True)
plt.savefig("przebieg_czasowy_vtk.png")
plt.close()

# Transformata Fouriera (FFT) z normalizacją
N = len(t)
yf_inc = np.abs(rfft(eta_inc)) * (2.0 / N)
yf_ref = np.abs(rfft(eta_ref)) * (2.0 / N)
yf_trans = np.abs(rfft(eta_down)) * (2.0 / N)
xf = rfftfreq(N, d=dt)

mask = (xf > 0)
f_freq = xf[mask]
amp_inc = yf_inc[mask]
amp_ref = yf_ref[mask]
amp_trans = yf_trans[mask]

# Filtracja dzielenia przez zero dla małych amplitud szumu
valid = amp_inc > (np.max(amp_inc) * 0.05)
Cr_f = np.zeros_like(amp_inc)
Ct_f = np.zeros_like(amp_inc)

Cr_f[valid] = amp_ref[valid] / amp_inc[valid]
Ct_f[valid] = amp_trans[valid] / amp_inc[valid]

# Przeliczenie osi na długość fali
c_wave = np.sqrt(0.00209424 * 9.55)
lambda_wave = c_wave / f_freq

# Wykres współczynników widmowych w paśmie badawczym
band_mask = (lambda_wave >= 150) & (lambda_wave <= 450)

plt.figure(figsize=(10, 6))
plt.plot(lambda_wave[band_mask], Cr_f[band_mask] * 100, label='Odbicie Cr(λ)', color='red', linewidth=2)
plt.plot(lambda_wave[band_mask], Ct_f[band_mask] * 100, label='Transmisja Ct(λ)', color='green', linewidth=2)
plt.axvspan(200, 400, color='gray', alpha=0.15, label='Zadane pasmo optymalizacji')
plt.axvline(302, color='black', linestyle=':', label='Rezonans układu Euve (~302 LU)')

plt.title("Widmowa skuteczność falochronu (Analiza FFT)")
plt.xlabel("Długość fali λ [LU]")
plt.ylabel("Współczynnik [%]")
plt.xlim(150, 450)
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig("skutecznosc_widmowa.png")
plt.close()

print("Analiza widmowa zakończona. Wygenerowano wykresy.")