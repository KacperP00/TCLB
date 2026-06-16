import os
import sys
import glob
import numpy as np
import pyvista as pv
from scipy.optimize import least_squares
import pandas as pd

# =============================================================================
# 1. KONFIGURACJA 
# =============================================================================
if len(sys.argv) < 2:
    raise ValueError("Błąd: Wymagany argument <lambda>. Użycie: python3 anal.py <lambda>")
current_lambda = float(sys.argv[1])

CASE_DIR = "output/"
VTK_PREFIX = "optimization_VTK_P00_"

GRAVITY = 0.00209424
HEIGHT = 9.55
C_WAVE = np.sqrt(GRAVITY * HEIGHT)
PERIOD_ITERS = int(np.round(current_lambda / C_WAVE))

Y_RANGE = (108, 208)

X_UPSTREAM = (400, 1800)
X_DOWNSTREAM = (2600, 4000)

X_START = 2000.0
X_END = 2400.0

# =============================================================================
# 2. FUNKCJE POMOCNICZE
# =============================================================================
def load_last_period_vtk(folder, prefix, period, y_range):
    search_pattern = os.path.join(folder, "**", f"*{prefix}*.vti")
    files = sorted(glob.glob(search_pattern, recursive=True))
    if not files:
        raise ValueError(f"Brak plików VTK dla wzorca: {search_pattern}")
        
    all_iters = [int(f.split('_')[-1].split('.')[0]) for f in files]
    max_it = max(all_iters)
    min_it = max_it - period
    
    valid_files = [f for f, it in zip(files, all_iters) if min_it <= it <= max_it]
    time_array = [it for it in all_iters if min_it <= it <= max_it]
    
    time_series = []
    x_coords = None
    y_min, y_max = y_range
    
    for f in valid_files:
        mesh = pv.read(f)
        rho_flat = mesh.get_array('Rho')
        is_cell_data = (rho_flat.size < np.prod(mesh.dimensions))
        
        if is_cell_data:
            actual_dims = [max(1, d - 1) for d in mesh.dimensions]
        else:
            actual_dims = mesh.dimensions
            
        rho = rho_flat.reshape(actual_dims, order='F')
        
        if x_coords is None:
            x_bounds = mesh.bounds[0:2]
            y_bounds = mesh.bounds[2:4]
            dims = actual_dims[0:2]
            
            offset_x = mesh.spacing[0]/2 if is_cell_data else 0
            offset_y = mesh.spacing[1]/2 if is_cell_data else 0
            
            x_coords = np.linspace(x_bounds[0], x_bounds[1], dims[0], endpoint=False) + offset_x
            y_coords = np.linspace(y_bounds[0], y_bounds[1], dims[1], endpoint=False) + offset_y
            y_mask = (y_coords >= y_min) & (y_coords <= y_max)
            
        if rho.ndim == 3:
            y_slice = rho[:, y_mask, 0]
        else:
            y_slice = rho[:, y_mask]
            
        time_series.append(np.mean(y_slice, axis=1))
        
    return x_coords, np.array(time_series), np.array(time_array)

def extract_fourier(time_series, time_array, period_iters):
    omega = 2.0 * np.pi / period_iters
    time_array = time_array - time_array[0]
    
    M = np.column_stack((
        np.cos(omega * time_array),
        np.sin(omega * time_array),
        np.ones_like(time_array)
    ))
    X_sol, _, _, _ = np.linalg.lstsq(M, time_series, rcond=None)
    complex_amp = X_sol[0] - 1j * X_sol[1]
    return X_sol[2], complex_amp, omega

def fit_wave_2way_stable(x_arr, complex_amp, k_guess):
    x0 = x_arr[0]
    x_loc = x_arr - x0
    
    def residuals(p):
        Af, phase_f, Ab, phase_b, kr, ki = p
        # NAPRAWIONA FIZYKA ZNAKÓW FAZOWYCH
        # Fala forward: exp(-1j * kr * x)
        # Fala backward: exp(+1j * kr * x)
        model = Af * np.exp(ki * x_loc - 1j * kr * x_loc) * np.exp(1j * phase_f) + \
                Ab * np.exp(-ki * x_loc + 1j * kr * x_loc) * np.exp(1j * phase_b)
        res = complex_amp - model
        return np.concatenate((np.real(res), np.imag(res)))
        
    A_guess = np.mean(np.abs(complex_amp))
    p0 = [A_guess, 0.0, 0.1 * A_guess, 0.0, k_guess, -0.0005]
    bounds = (
        [0, -np.pi, 0, -np.pi, k_guess * 0.8, -0.01],
        [A_guess * 5, np.pi, A_guess * 5, np.pi, k_guess * 1.2, 0.0]
    )
    
    res = least_squares(residuals, p0, bounds=bounds)
    Af, phase_f, Ab, phase_b, kr, ki = res.x
    
    Af_loc = Af * np.exp(1j * phase_f)
    Ab_loc = Ab * np.exp(1j * phase_b)
    
    model = Af * np.exp(ki * x_loc - 1j * kr * x_loc) * np.exp(1j * phase_f) + \
            Ab * np.exp(-ki * x_loc + 1j * kr * x_loc) * np.exp(1j * phase_b)
            
    ss_res = np.sum(np.abs(complex_amp - model)**2)
    ss_tot = np.sum(np.abs(complex_amp - np.mean(complex_amp))**2)
    r2 = max(0.0, 1 - (ss_res / ss_tot))
    
    return Af_loc, Ab_loc, kr, ki, r2

# =============================================================================
# 3. GŁÓWNA ANALIZA
# =============================================================================
if __name__ == "__main__":
    print(f"\n--- ANALIZA KINEMATYCZNA DLA LAMBDA = {current_lambda} ---")
    
    x, t_series, t_array = load_last_period_vtk(CASE_DIR, VTK_PREFIX, PERIOD_ITERS, Y_RANGE)
    _, complex_amp, omega = extract_fourier(t_series, t_array, PERIOD_ITERS)
    
    mask_up = (x >= X_UPSTREAM[0]) & (x <= X_UPSTREAM[1])
    mask_down = (x >= X_DOWNSTREAM[0]) & (x <= X_DOWNSTREAM[1])
    
    k_guess = omega / C_WAVE
    
    Af_loc_up, Ab_loc_up, kr_up, ki_up, r2_up = fit_wave_2way_stable(x[mask_up], complex_amp[mask_up], k_guess)
    Af_loc_down, Ab_loc_down, kr_down, ki_down, r2_down = fit_wave_2way_stable(x[mask_down], complex_amp[mask_down], k_guess)
    
    # Przesunięcie fal Upstream na wejście do wnęki (X=2000)
    dist_in = X_START - x[mask_up][0]
    A_in = Af_loc_up * np.exp(ki_up * dist_in) * np.exp(-1j * kr_up * dist_in)
    
    # Propagacja fali odbitej powrotem DO źródła wnęki (uzyskuje energię na dystansie)
    Ab_cavity = Ab_loc_up * np.exp(-ki_up * dist_in) * np.exp(1j * kr_up * dist_in)
    
    # Przesunięcie fali Downstream na wyjście z wnęki (X=2400) - dystans ujemny
    dist_out = X_END - x[mask_down][0]
    A_out = Af_loc_down * np.exp(ki_down * dist_out) * np.exp(-1j * kr_down * dist_out)
    
    # Obliczenie współczynników inżynierskich bezbłędnie w osiach wnęk
    R_coef = abs(Ab_cavity) / abs(A_in)
    T_coef = abs(A_out) / abs(A_in)        
    D_coef = 1.0 - (T_coef**2) - (R_coef**2)
    
    print("\n" + "="*50)
    print(" WYNIKI ANALIZY KINEMATYCZNEJ WNĘK")
    print("="*50)
    print(f"Jakość dopasowania (Upstream R2)   : {r2_up:.4f}")
    print(f"Jakość dopasowania (Downstream R2) : {r2_down:.4f}")
    print(f"Amplituda padająca na wnękę (A_in) : {abs(A_in):.5f}")
    print(f"Amplituda wychodząca (A_out)       : {abs(A_out):.5f}")
    print("-" * 50)
    print(f"Wsp. Transmisji (K_T)              : {T_coef*100:.2f} %")
    print(f"Wsp. Odbicia (K_R) przy wnęce      : {R_coef*100:.2f} %")
    print(f"Dyssypacja (Strata Energii)        : {D_coef*100:.2f} %")
    print("="*50)
    
    output_file = 'kp/shallow_water/thesis/wyniki_wneki.csv'
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, 'a') as f:
        if not file_exists:
            f.write("lambda,K_T,K_R,Dissipation,A_in,A_out,R2_up,R2_down\n")
        f.write(f"{current_lambda:.1f},{T_coef:.6f},{R_coef:.6f},{D_coef:.6f},{abs(A_in):.6f},{abs(A_out):.6f},{r2_up:.4f},{r2_down:.4f}\n")

    import matplotlib.pyplot as plt

    # Rysowanie rzeczywistej amplitudy wyciągniętej z symulacji
    plt.figure(figsize=(12, 6))
    
    # Surowe dane wyciągnięte przez Fouriera
    plt.plot(x, np.abs(complex_amp), color='lightgray', label='Surowa Amplituda (Fourier)')
    
    # Dane w strefach pomiarowych
    plt.plot(x[mask_up], np.abs(complex_amp[mask_up]), 'b.', markersize=4, label='Strefa Pomiaru (Upstream)')
    plt.plot(x[mask_down], np.abs(complex_amp[mask_down]), 'r.', markersize=4, label='Strefa Pomiaru (Downstream)')
    
    # Oznaczenie wnęk
    plt.axvline(X_START, color='k', linestyle='--', label='Początek Wnęk')
    plt.axvline(X_END, color='k', linestyle='--', label='Koniec Wnęk')
    
    plt.title(f"Rozkład amplitudy fali w kanale dla lambda = {current_lambda}")
    plt.xlabel("Pozycja w kanale (X)")
    plt.ylabel("Amplituda fali")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Zapis do pliku
    plt.savefig(f"profil_amplitudy_{current_lambda:.0f}.png", dpi=300)
    print(f"Zapisano wykres profilu fali do pliku: profil_amplitudy_{current_lambda:.0f}.png")