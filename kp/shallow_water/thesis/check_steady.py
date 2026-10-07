import pyvista as pv
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Konfiguracja
input_dir = "/mnt/d/TCLB/data_for_comparison/singlewave_euve_output/" # Zmień folder by sprawdzić konkretny model
out_prefix = "kp/shallow_water/thesis/wyniki_analizy_comparison/weryfikacja_steady_state"
os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

y_center = 154
h_base = 9.55
x_probe_up = 2500   # Punkt w strefie odbicia
x_probe_down = 6000 # Punkt w strefie transmisji

files = sorted(glob.glob(os.path.join(input_dir, "*.vti")))
if not files:
    files = sorted(glob.glob(os.path.join(input_dir, "*.pvti")))

print(f"Śledzenie historii czasowej przez {len(files)} plików...")

iterations = []
eta_up = []
eta_down = []

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

# Wykres surowego wychylenia wody w czasie
plt.figure(figsize=(12, 6))
plt.plot(t, eta_up, label='Sygnał Upstream (Odbicie + Padająca) X=2500', color='blue', alpha=0.8)
plt.plot(t, eta_down, label='Sygnał Downstream (Transmisja) X=6000', color='green', alpha=0.8)

# Zaznaczenie strefy analizowanej przez 2D FFT (ostatnie 250 plików, zakładając zapis co np. 100 iter)
t_fft_start = t[-250] if len(t) >= 250 else t[0]
plt.axvspan(t_fft_start, t[-1], color='gray', alpha=0.3, label='Okno Analizy 2D FFT')

plt.title("Weryfikacja Stanu Ustalonego (Steady-State Evolution)")
plt.xlabel("Iteracja")
plt.ylabel("Wychylenie [LU]")
plt.legend()
plt.grid(True)
plt.savefig(f"{out_prefix}_timeline.png", dpi=300)
plt.close()

print(f"Zapisano wykres weryfikacyjny: {out_prefix}_timeline.png")