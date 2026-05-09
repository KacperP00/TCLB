import os
import sys
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Konfiguracja ścieżek
CASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "output/"
VTK_PREFIX = "01clean_canal_VTK_P00_"
PLOT_DIR = "wykresy/"

# Parametry fizyczne i czasowe
LAMBDA_LU = 650.0
PERIOD_ITERS = 4596.19
VTK_STEP = 50
MIN_ITER = 25000
MAX_ITER = 38800

# Geometria stref pomiarowych (wyizolowany środek kanału)
Y_RANGE = (160, 192) 
X_UPSTREAM = (3000, 3700) 
X_DOWNSTREAM = (5000, 6700)

def load_and_average_vtk(folder, prefix, min_iter, y_range, x_range_full):
    # Wczytanie i agregacja danych z plików VTK
    search_pattern = os.path.join(folder, f"*{prefix}*.vti")
    files = sorted(glob.glob(search_pattern))
    
    valid_files = [f for f in files if min_iter <= int(f.split('_')[-1].split('.')[0]) <= MAX_ITER]
    
    if not valid_files:
        raise FileNotFoundError("Brak plików w zadanym oknie czasowym.")

    first_mesh = pv.read(valid_files[0]).cell_data_to_point_data()
    nx = first_mesh.dimensions[0]
    
    x_start, x_end = x_range_full
    x_coords = np.arange(x_start, x_end)
    data_matrix = np.zeros((len(valid_files), len(x_coords)))
    time_steps = []
    
    print(f"[Wczytywanie] Przetwarzanie {len(valid_files)} plików...")
    
    for t_idx, filepath in enumerate(valid_files):
        iteration = int(filepath.split('_')[-1].split('.')[0])
        time_steps.append(iteration)
        
        mesh = pv.read(filepath).cell_data_to_point_data()
        dims = mesh.dimensions
        
        rho_3d = mesh.point_data["Rho"].reshape((dims[2], dims[1], dims[0]))
        rho_2d = rho_3d[0, :, :]
        
        # Uśrednianie tylko w zadanym przedziale osi Y
        rho_averaged = np.mean(rho_2d[y_range[0]:y_range[1], x_start:x_end], axis=0)
        data_matrix[t_idx, :] = rho_averaged
        
    return data_matrix, np.array(time_steps), x_coords

def temporal_transform(data_matrix, time_steps, omega, dt):
    # Ekstrakcja amplitudy i fazy fali z wykorzystaniem okna Hanninga
    complex_amps = []
    t_window = time_steps[-1] - time_steps[0]
    
    window = np.hanning(len(time_steps))
    window_correction = 2.0 
    
    for x_idx in range(data_matrix.shape[1]):
        rho_1d = data_matrix[:, x_idx]
        rho_centered = rho_1d - np.mean(rho_1d)
        
        signal = rho_centered * window * np.exp(-1j * omega * time_steps)
        integral = np.trapezoid(signal, x=time_steps)
        
        coeff = (integral / t_window) * 2.0 * window_correction
        complex_amps.append(coeff)
        
    return np.array(complex_amps)

def residuals_upstream(params, x, eta_obs):
    # Funkcja reszt dla modelu fali padającej i odbitej
    ReA, ImA, ReB, ImB, kr, ki = params
    A = ReA + 1j * ImA
    B = ReB + 1j * ImB
    k_complex = kr + 1j * ki
    
    model = A * np.exp(-1j * k_complex * x) + B * np.exp(1j * k_complex * x)
    diff = model - eta_obs
    return np.concatenate([diff.real, diff.imag])

def residuals_downstream_fixed_k(params, x, eta_obs, k_complex):
    # Funkcja reszt dla modelu fali przechodzącej (ustalony wektor k)
    ReC, ImC = params
    C = ReC + 1j * ImC
    
    model = C * np.exp(-1j * k_complex * x)
    diff = model - eta_obs
    return np.concatenate([diff.real, diff.imag])

def calc_stats(res, x_coords, eta_vals):
    # Obliczanie metryk dopasowania (R^2, RMSE)
    sse = np.sum(res.fun**2)
    vals_stacked = np.concatenate([eta_vals.real, eta_vals.imag])
    sst = np.sum((vals_stacked - np.mean(vals_stacked))**2)
    fit_quality = 1 - (sse / sst) if sst > 0 else 0
    rmse = np.sqrt(sse / (2 * len(x_coords)))
    return fit_quality, rmse

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"\n--- Analiza Odbicia i Transmisji ---")
    
    full_x_range = (X_UPSTREAM[0], X_DOWNSTREAM[1])
    data_matrix, time_steps, x_coords = load_and_average_vtk(
        CASE_DIR, VTK_PREFIX, MIN_ITER, Y_RANGE, full_x_range
    )
    
    # Globalny poziom tła do kompensacji Mass Drift w modelu C
    mean_water_level = np.mean(data_matrix)
    
    omega = 2.0 * np.pi / PERIOD_ITERS
    dt = VTK_STEP
    eta_vals = temporal_transform(data_matrix, time_steps, omega, dt)
    k_theory = (2.0 * np.pi) / LAMBDA_LU
    
    # Analiza strefy dolotowej (Upstream)
    mask_up = (x_coords >= X_UPSTREAM[0]) & (x_coords < X_UPSTREAM[1])
    x_up, eta_up = x_coords[mask_up], eta_vals[mask_up]
    
    bounds_up = ([-np.inf, -np.inf, -np.inf, -np.inf, k_theory * 0.8, -0.01], 
                 [np.inf, np.inf, np.inf, np.inf, k_theory * 1.2, 0.01])
    x0_up = [np.mean(np.abs(eta_up)), 0.0, 0.0, 0.0, k_theory, 0.0]
    
    res_up = least_squares(residuals_upstream, x0_up, args=(x_up, eta_up), bounds=bounds_up)
    ReA, ImA, ReB, ImB, kr_up, ki_up = res_up.x
    A = ReA + 1j * ImA
    B = ReB + 1j * ImB
    k_comp_up = kr_up + 1j * ki_up
    
    fit_q_up, rmse_up = calc_stats(res_up, x_up, eta_up)
    
    # Analiza strefy wylotowej (Downstream)
    mask_down = (x_coords >= X_DOWNSTREAM[0]) & (x_coords < X_DOWNSTREAM[1])
    x_down, eta_down = x_coords[mask_down], eta_vals[mask_down]
    
    x0_down = [np.mean(np.abs(eta_down)), 0.0]
    res_down = least_squares(residuals_downstream_fixed_k, x0_down, args=(x_down, eta_down, k_comp_up))
    
    ReC, ImC = res_down.x
    C = ReC + 1j * ImC
    
    kr_down, ki_down = kr_up, ki_up 
    k_comp_down = k_comp_up
    
    fit_q_down, rmse_down = calc_stats(res_down, x_down, eta_down)
    
    # Przeliczenie amplitud na krawędź ewentualnej przeszkody
    X_START = 3900
    
    A_inc = abs(A) * np.exp(ki_up * X_START)
    A_ref = abs(B) * np.exp(-ki_up * X_START)
    A_trans = abs(C) * np.exp(ki_up * X_START)

    R_val = A_ref / A_inc if A_inc > 0 else 0
    T_val = A_trans / A_inc if A_inc > 0 else 0
    D_val = 1.0 - (R_val**2 + T_val**2)

    # Wydruk w konsoli
    print("\n[1] STREFA DOLOTOWA")
    print(f"Fala Padająca (A): {abs(A):.6f} lu (Faza: {np.angle(A, deg=True):.1f}°)")
    print(f"Fala Odbita (B):   {abs(B):.6f} lu (Faza: {np.angle(B, deg=True):.1f}°)")
    print(f"Liczba falowa k:   {kr_up:.5f} + {ki_up:.2e}j")
    print(f"Jakość Fit (R^2):  {fit_q_up:.4f} | RMSE: {rmse_up:.6f}")
    
    print("\n[2] WYNIKI FIZYCZNE (Granice wnęk)")
    print(f"Amplituda padająca (X={X_START}):      {A_inc:.6f} lu")
    print(f"Amplituda odbita (X={X_START}):        {A_ref:.6f} lu")
    print("-" * 35)
    print(f"=> Odbicie (R):    {R_val:.4f}")
    print(f"=> Transmisja (T): {T_val:.4f}")
    print(f"=> Dyssypacja (D): {D_val:.4f}")
    print(f"Średni poziom wody: {mean_water_level:.6f}")

    print("\n[3] GOTOWE PARAMETRY DO PLIKU XML")
    print("Skopiuj poniższe linijki bezpośrednio do sekcji <Model> w pliku .xml:")
    print("-" * 55)
    print(f'        <Param name="Height" value="{mean_water_level:.6f}"/>')
    print(f'        <Param name="Wave_A" value="{abs(A):.6f}"/>')
    print(f'        <Param name="Wave_k_real" value="{kr_up:.6f}"/>')
    print(f'        <Param name="Wave_k_imag" value="{ki_up:.6f}"/>')
    print(f'        <Param name="Wave_w" value="{omega:.6f}"/>')
    print(f'        <Param name="Wave_Phase" value="{-np.angle(A):.6f}"/>')
    print("-" * 55)

    # Generowanie wykresów
    plot_filename = os.path.join(PLOT_DIR, f"{VTK_PREFIX}analysis.png")
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharey='row')
    
    model_up = A * np.exp(-1j * k_comp_up * x_up) + B * np.exp(1j * k_comp_up * x_up)
    model_down = C * np.exp(-1j * k_comp_down * x_down)
    
    axs[0,0].plot(x_up, eta_up.real, 'ko', alpha=0.4, markersize=3, label='Symulacja Re')
    axs[0,0].plot(x_up, model_up.real, 'r-', linewidth=1.5, label='Model Re')
    axs[0,0].set_title(f"Dolot (Re) | R = {R_val:.4f}")
    axs[0,0].set_ylabel(r"$Re(\eta)$ [lu]")
    axs[0,0].grid(True, alpha=0.3); axs[0,0].legend()
    
    axs[1,0].plot(x_up, eta_up.imag, 'bo', alpha=0.4, markersize=3, label='Symulacja Im')
    axs[1,0].plot(x_up, model_up.imag, 'r-', linewidth=1.5, label='Model Im')
    axs[1,0].set_title("Dolot (Im)")
    axs[1,0].set_ylabel(r"$Im(\eta)$ [lu]"); axs[1,0].set_xlabel("X [lu]")
    axs[1,0].grid(True, alpha=0.3); axs[1,0].legend()
    
    axs[0,1].plot(x_down, eta_down.real, 'ko', alpha=0.4, markersize=3, label='Symulacja Re')
    axs[0,1].plot(x_down, model_down.real, 'g-', linewidth=1.5, label='Model Re')
    axs[0,1].set_title(f"Wylot (Re) | T = {T_val:.4f}")
    axs[0,1].grid(True, alpha=0.3); axs[0,1].legend()
    
    axs[1,1].plot(x_down, eta_down.imag, 'bo', alpha=0.4, markersize=3, label='Symulacja Im')
    axs[1,1].plot(x_down, model_down.imag, 'g-', linewidth=1.5, label='Model Im')
    axs[1,1].set_title("Wylot (Im)")
    axs[1,1].set_xlabel("X [lu]")
    axs[1,1].grid(True, alpha=0.3); axs[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    print(f"\n[Wykres] Zapisano analizę do: {plot_filename}")

if __name__ == "__main__":
    main()