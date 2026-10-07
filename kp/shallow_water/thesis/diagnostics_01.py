import pyvista as pv
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# Konfiguracja ścieżek
input_dir = "output"
out_prefix = "kp/shallow_water/thesis/wyniki_analizy_comparison/wyniki_euve_basic"
os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

# PARAMETRY PODSTAWOWEJ SIATKI
y_center = 154
x_probe_up = 2500      # Sonda przed przeszkodą (pozostaje bezpieczna)
x_probe_down = 4800    # PRZESUNIĘTA SONDA - ratuje uciętą falę!
h_base = 9.55
c_wave = np.sqrt(0.00209424 * h_base)
split_iter = 43000     # Optymalny punkt podziału dla X=2500

files = sorted(glob.glob(os.path.join(input_dir, "*.vti")))
if not files:
    files = sorted(glob.glob(os.path.join(input_dir, "*.pvti")))

iterations = []
eta_up = []
eta_down = []

print("Ekstrakcja przebiegów czasowych (Podstawowy euve)...")
for i, f in enumerate(files):
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
    
    dist_up = np.sqrt((x_coords - x_probe_up)**2 + (y_coords - y_center)**2)
    dist_down = np.sqrt((x_coords - x_probe_down)**2 + (y_coords - y_center)**2)
    
    eta_up.append(rho[np.argmin(dist_up)] - h_base)
    eta_down.append(rho[np.argmin(dist_down)] - h_base)

t = np.array(iterations)
eta_up = np.array(eta_up)
eta_down = np.array(eta_down)
dt = np.mean(np.diff(t))

# Separacja impulsów na wlocie
eta_inc = np.copy(eta_up)
eta_ref = np.copy(eta_up)

mask_ref = t > split_iter
eta_inc[mask_ref] = 0.0
eta_ref[~mask_ref] = 0.0

# Generowanie wykresu kontrolnego (sprawdź wizualnie, czy zielony impuls opada do 0 przed 65k!)
plt.figure(figsize=(12, 6))
plt.plot(t, eta_inc, label=f'Padająca (X={x_probe_up})', color='blue')
plt.plot(t, eta_ref, label=f'Odbita (X={x_probe_up})', color='red')
plt.plot(t, eta_down, label=f'Transmisja (X={x_probe_down})', color='green')
plt.axvline(split_iter, color='k', linestyle='--', label='Bariera separacji')
plt.xlabel("Iteracja")
plt.ylabel("Wychylenie [LU]")
plt.title("Przebieg czasowy impulsów fali - euve (Podstawowy)")
plt.legend()
plt.grid(True)
plt.savefig(f"{out_prefix}_czas.png", dpi=300)
plt.close()

# Transformaty Fouriera
N = len(t)
yf_inc = np.abs(rfft(eta_inc)) * (2.0 / N)
yf_ref = np.abs(rfft(eta_ref)) * (2.0 / N)
yf_trans = np.abs(rfft(eta_down)) * (2.0 / N)
xf = rfftfreq(N, d=dt)

mask_freq = (xf > 0)
f_freq = xf[mask_freq]
amp_inc = yf_inc[mask_freq]
amp_ref = yf_ref[mask_freq]
amp_trans = yf_trans[mask_freq]

valid = amp_inc > (np.max(amp_inc) * 0.05)
Cr_f = np.zeros_like(amp_inc)
Ct_f = np.zeros_like(amp_inc)

Cr_f[valid] = amp_ref[valid] / amp_inc[valid]
Ct_f[valid] = amp_trans[valid] / amp_inc[valid]

# Energia
D_energy = np.zeros_like(amp_inc)
D_energy[valid] = 1.0 - (Cr_f[valid]**2) - (Ct_f[valid]**2)

lambda_wave = c_wave / f_freq

df_spectrum = pd.DataFrame({
    'Czestotliwosc': f_freq,
    'Dlugosc_Fali': lambda_wave,
    'Amp_Inc': amp_inc,
    'Amp_Ref': amp_ref,
    'Amp_Trans': amp_trans,
    'Cr_Amp': Cr_f,
    'Ct_Amp': Ct_f,
    'D_Energy': D_energy
})
df_spectrum.to_csv(f"{out_prefix}.csv", index=False)

band_mask = (lambda_wave >= 150) & (lambda_wave <= 450)

plt.figure(figsize=(10, 6))
plt.plot(lambda_wave[band_mask], Cr_f[band_mask] * 100, label='Odbicie Cr(λ)', color='red', linewidth=2)
plt.plot(lambda_wave[band_mask], Ct_f[band_mask] * 100, label='Transmisja Ct(λ)', color='green', linewidth=2)
plt.axvspan(200, 400, color='gray', alpha=0.15, label='Pasmo (200-400 LU)')

plt.title("Widmo - euve (Basic)")
plt.xlabel("Długość fali λ [LU]")
plt.ylabel("Współczynnik [%]")
plt.xlim(150, 450)
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig(f"{out_prefix}.png", dpi=300)
plt.close()

band_strict = (lambda_wave >= 200) & (lambda_wave <= 400) & valid
mean_Cr = np.mean(Cr_f[band_strict]) * 100
mean_Ct = np.mean(Ct_f[band_strict]) * 100
mean_D = np.mean(D_energy[band_strict]) * 100

report_text = (
    "RAPORT Z ANALIZY WIDMOWEJ - EUVE (BASIC)\n"
    "================================================\n"
    f"Średnie odbicie amplitudowe (200-400 LU): {mean_Cr:.2f}%\n"
    f"Średnia transmisja amplitudowa (200-400 LU): {mean_Ct:.2f}%\n"
    f"Dyssypacja (Tłumienie lepkosciowe w wodzie): {mean_D:.2f}%\n"
)

with open(f"{out_prefix}.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)