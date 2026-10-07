import pyvista as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys

# Konfiguracja bezwzględnej ścieżki dostępu do wyników
output_dir = "output"

# Wyszukiwanie plików wolumetrycznych VTK
files = sorted(glob.glob(os.path.join(output_dir, "*.vti")))
if not files:
    files = sorted(glob.glob(os.path.join(output_dir, "*.pvti")))

if not files:
    print(f"Błąd: Brak plików danych w katalogu {output_dir}")
    sys.exit(1)

# Parametry referencyjne modelu (pozostają bez zmian)
h_base = 9.55
y_center = 154

# Inicjalizacja formatu danych przestrzennych
mesh_init = pv.read(files[0])
data_type = "cell_data" if "Rho" in mesh_init.cell_data else "point_data"

if data_type == "cell_data":
    x_coords = mesh_init.cell_centers().points[:, 0]
    y_coords = mesh_init.cell_centers().points[:, 1]
    rho_ref = mesh_init.cell_data["Rho"]
else:
    x_coords = mesh_init.points[:, 0]
    y_coords = mesh_init.points[:, 1]
    rho_ref = mesh_init.point_data["Rho"]

# Alokacja pamięci dla wektora maksymalnego wychylenia
max_envelope = np.zeros_like(rho_ref, dtype=float)

# Generowanie koperty fali po wszystkich iteracjach czasowych
for f in files:
    mesh = pv.read(f)
    rho = mesh.cell_data["Rho"] if data_type == "cell_data" else mesh.point_data["Rho"]
    eta = np.abs(rho - h_base)
    max_envelope = np.maximum(max_envelope, eta)

# Izolacja jednowymiarowego profilu na osi symetrii kanału
center_mask = np.abs(y_coords - y_center) < 1.5
x_profile = x_coords[center_mask]
env_profile = max_envelope[center_mask]

# Zapewnienie monotoniczności osi X
sort_idx = np.argsort(x_profile)
x_profile = x_profile[sort_idx]
env_profile = env_profile[sort_idx]

# ZMIANA 1: Nowa nazwa pliku CSV, aby nie nadpisać wyników topologii
csv_data = np.column_stack((x_profile, env_profile))
np.savetxt("kp/shallow_water/thesis/wyniki_walidacji/koperta_dane_euve.csv", csv_data, delimiter=",", header="X_Position,Max_Amplitude", comments="")
print("Zapisano dane wektorowe: koperta_dane_euve.csv")

# Izolacja strefy analizy fali stojącej przed falochronem (zostaje bez zmian, bo to ten sam obszar)
mask_win = (x_profile > 3000) & (x_profile < 3800)
x_win = x_profile[mask_win]
env_win = env_profile[mask_win]

# Obliczanie i eliminacja numerycznego tłumienia lepkiego wody
z = np.polyfit(x_win, env_win, 2)
p = np.poly1d(z)
trend = p(x_win)
detrended = env_win - trend

# ZMIANA 2: Przesunięcie punktu ekstrapolacji bliżej krawędzi wnęk Euve (X = 4154)
A_standing = (np.max(detrended) - np.min(detrended)) / 2.0
A_inc_local = p(4150) 

# Kalkulacja rygorystycznych współczynników materiałowych
C_R = A_standing / A_inc_local

mask_after = (x_profile > 4600) & (x_profile < 7500)
A_trans = np.mean(env_profile[mask_after])
C_T = A_trans / A_inc_local

# Rysowanie skorygowanego profilu
plt.figure(figsize=(12, 6))
plt.plot(x_profile, env_profile, color='blue', linewidth=1.5, label='Koperta fali')

# ZMIANA 3: Nowe pole szarego tła odpowiadające fizycznym wymiarom rezonatorów Euve (X: 4154 -> 4246)
plt.axvspan(4154, 4246, color='gray', alpha=0.3, label='Rezonatory Autlera-Townesa')

plt.axhline(A_inc_local, color='red', linestyle=':', alpha=0.6, label='Fala padająca (przy przeszkodzie)')
plt.axhline(A_trans, color='green', linestyle='--', alpha=0.8, label='Transmisja')

# Pole z wynikami analizy
info_text = (
    f"Amplituda padająca (Ao): {A_inc_local:.4f}\n"
    f"Odbicie (Cr): {C_R:.2%}\n"
    f"Transmisja (Ct): {C_T:.2%}"
)
plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Formatowanie przestrzeni roboczej wykresu
plt.xlabel('Pozycja X [węzły]')
plt.ylabel('Amplituda bezwzględna |h - h_base|')
plt.title('Profil tłumienia fali z korektą trendu lepkiego (Model Referencyjny Euve)')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# ZMIANA 4: Nowa nazwa pliku PNG
plt.savefig('kp/shallow_water/thesis/wyniki_walidacji/koperta_wspolczynniki_euve.png', dpi=300)
print(f"Rzeczywiste Cr = {C_R:.2%} | Rzeczywiste Ct = {C_T:.2%}")
print("Zapisano wizualizację: koperta_wspolczynniki_euve.png")