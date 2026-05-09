import os
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Konfiguracja ścieżek
CASE_DIR = "output/"
VTK_PREFIX = "02empty_cav_VTK_P00_"
PLOT_DIR = "wykresy/"

# Parametry fizyczne i numeryczne
LAMBDA_LU = 320.0
PERIOD_ITERS = 2263
VTK_STEP = 50
MIN_ITER = 11315

# Strefa pomiarowa
# Geometria stref pomiarowych
Y_RANGE = (336, 656) # Światło głównego kanału z wodą
X_UPSTREAM = (400, 1000)

def load_and_average_vtk(folder, prefix, min_iter, y_range, x_range):
    # Wyszukiwanie plików wynikowych
    search_pattern = os.path.join(folder, f"*{prefix}*.vti")
    files = sorted(glob.glob(search_pattern))
    valid_files = [f for f in files if int(f.split('_')[-1].split('.')[0]) >= min_iter]
    
    if not valid_files:
        raise FileNotFoundError("Brak plików w podanym katalogu.")

    # Pobranie wymiarów domeny 3D (X, Y, Z) z pierwszego pliku
    first_mesh = pv.read(valid_files[0]).cell_data_to_point_data()
    nx = first_mesh.dimensions[0]
    
    # Ograniczenie indeksów do strefy pomiarowej
    x_start, x_end = x_range
    x_coords = np.arange(x_start, x_end)
    data_matrix = np.zeros((len(valid_files), len(x_coords)))
    time_steps = []
    
    print(f"[Wczytywanie] Znaleziono {len(valid_files)} plików. Przetwarzanie...")
    
    for t_idx, filepath in enumerate(valid_files):
        iteration = int(filepath.split('_')[-1].split('.')[0])
        time_steps.append(iteration)
        
        mesh = pv.read(filepath).cell_data_to_point_data()
        dims = mesh.dimensions
        
        # Ekstrakcja dolnej płaszczyzny 2D (z=0)
        rho_3d = mesh.point_data["Rho"].reshape((dims[2], dims[1], dims[0]))
        rho_2d = rho_3d[0, :, :]
        
        # Całkowanie sygnału do postaci 1D dla zadanego wycinka X
        rho_averaged = np.mean(rho_2d[y_range[0]:y_range[1], x_start:x_end], axis=0)
        data_matrix[t_idx, :] = rho_averaged
        
    return data_matrix, np.array(time_steps), x_coords

def temporal_transform(data_matrix, time_steps, omega, dt):
    # Całkowanie czasowe sygnału 
    complex_amps = []
    t = time_steps
    t_f = t[-1] - t[0]
    
    for x_idx in range(data_matrix.shape[1]):
        eta = data_matrix[:, x_idx]
        eta_centered = eta - np.mean(eta)
        
        # Ujemny znak synchronizuje konwencję kierunku propagacji (A=padająca, B=odbita)
        integral = np.sum(eta_centered * np.exp(-1j * omega * t)) * dt
        coeff = (integral / t_f) * 2.0
        complex_amps.append(coeff)
        
    return np.array(complex_amps)

def residuals(params, x, eta_obs):
    # Model z zespoloną liczbą falową (k = kr + i*ki)
    ReA, ImA, ReB, ImB, kr, ki = params
    
    A = ReA + 1j * ImA
    B = ReB + 1j * ImB
    k_complex = kr + 1j * ki
    
    # Superpozycja fali padającej i odbitej
    model = A * np.exp(-1j * k_complex * x) + B * np.exp(1j * k_complex * x)
    
    diff = model - eta_obs
    return np.concatenate([diff.real, diff.imag])

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"\n--- Analiza Odbicia (Strefa Dolotowa) ---")
    
    # 1. Pobranie i przygotowanie danych
    data_matrix, time_steps, x_coords = load_and_average_vtk(
        CASE_DIR, VTK_PREFIX, MIN_ITER, Y_RANGE, X_UPSTREAM
    )
    
    # 2. Rzutowanie Fouriera na zespoloną amplitudę przestrzenną
    omega = 2.0 * np.pi / PERIOD_ITERS
    dt = VTK_STEP
    eta_vals = temporal_transform(data_matrix, time_steps, omega, dt)
    
    # 3. Dopasowanie modelu (Fit)
    k_theory = (2.0 * np.pi) / LAMBDA_LU
    
    # Granice optymalizatora: zezwalamy na tłumienie (ki)
    bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, k_theory * 0.8, -0.01], 
              [np.inf, np.inf, np.inf, np.inf, k_theory * 1.2, 0.01])
    
    amp_guess = np.mean(np.abs(eta_vals))
    x0 = [amp_guess, 0.0, 0.0, 0.0, k_theory, 0.0]
    
    res = least_squares(residuals, x0, args=(x_coords, eta_vals), bounds=bounds)
    
    # 4. Interpretacja wyników optymalizacji
    ReA, ImA, ReB, ImB, kr_fit, ki_fit = res.x
    A = ReA + 1j * ImA # Fala padająca
    B = ReB + 1j * ImB # Fala odbita
    k_complex_fit = kr_fit + 1j * ki_fit
    
    R_complex = B / A if abs(A) > 0 else 0j
    R_val = abs(R_complex)
    
    # Statystyki dopasowania
    sse = np.sum(res.fun**2)
    vals_stacked = np.concatenate([eta_vals.real, eta_vals.imag])
    sst = np.sum((vals_stacked - np.mean(vals_stacked))**2)
    fit_quality = 1 - (sse / sst) if sst > 0 else 0
    rmse = np.sqrt(sse / (2 * len(x_coords)))

    # --- WYDRUK KONTROLNY ---
    print("\n[WYNIKI DOPASOWANIA]")
    print(f"Fala Padająca (A): {abs(A):.6f} lu (Faza: {np.angle(A, deg=True):.1f}°)")
    print(f"Fala Odbita (B):   {abs(B):.6f} lu (Faza: {np.angle(B, deg=True):.1f}°)")
    print(f"Liczba falowa k:   {kr_fit:.5f} + {ki_fit:.2e}j")
    print(f"Tłumienie fali:    Zidentyfikowano (ki = {ki_fit:.2e})")
    print(f"Jakość Fit (R^2):  {fit_quality:.4f}")
    print(f"Błąd RMSE:         {rmse:.6f}")
    
    print(f"\n=> WSPÓŁCZYNNIK ODBICIA (R): {R_val:.4f} <=")

    # --- WIZUALIZACJA ---
    plot_filename = os.path.join(PLOT_DIR, "fit_upstream_ReIm.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    model_complex = A * np.exp(-1j * k_complex_fit * x_coords) + B * np.exp(1j * k_complex_fit * x_coords)
    
    # Część Rzeczywista
    ax1.plot(x_coords, eta_vals.real, 'ko', label='Symulacja Re', alpha=0.5, markersize=4)
    ax1.plot(x_coords, model_complex.real, 'r-', linewidth=2, label='Model Re')
    ax1.set_ylabel(r"$Re(\eta)$ [lu]")
    ax1.set_title(f"Strefa Dolotowa | Współczynnik Odbicia R={R_val:.4f} (FitQ={fit_quality:.4f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Część Urojona
    ax2.plot(x_coords, eta_vals.imag, 'bo', label='Symulacja Im', alpha=0.5, markersize=4)
    ax2.plot(x_coords, model_complex.imag, 'r-', linewidth=2, label='Model Im')
    ax2.set_xlabel("X [lu]")
    ax2.set_ylabel(r"$Im(\eta)$ [lu]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"\n[Wykres] Zapisano analizę do: {plot_filename}")

if __name__ == "__main__":
    main()