import os
import sys
import glob
import argparse
import numpy as np
import pyvista as pv
from scipy.optimize import least_squares

# =============================================================================
# 1. KONFIGURACJA Z ARGUMENTÓW LINII KOMEND
# =============================================================================
parser = argparse.ArgumentParser(description="Analiza fali w absorberze (T, R, D)")
parser.add_argument("--period", type=float, required=True, help="Okres fali w iteracjach")
parser.add_argument("--gravity", type=float, default=0.00209424, help="Grawitacja numeryczna")
parser.add_argument("--height", type=float, default=9.55, help="Głębokość numeryczna")
parser.add_argument("--case_dir", type=str, default="output/", help="Katalog VTK")
parser.add_argument("--csv", type=str, default="wyniki_absorbera.csv", help="Plik CSV z wynikami")
args = parser.parse_args()

VTK_PREFIX = "clean_canal_VTK_P00_"

# Fizyka i parametry narzucone z argumentów
C_WAVE = np.sqrt(args.gravity * args.height)
LAMBDA_LU = args.period * C_WAVE
PERIOD_ITERS = int(np.round(args.period))
Y_RANGE = (4, 104)

# Granice strefy absorbera (DesignSpace)
x_start = 2000.0
x_end = 2400.0

# Dynamiczne okna przestrzenne omijające ściany (Szerokość 1 lambda, odległość 0.5 lambda)
x_up_end = x_start - 0.5 * LAMBDA_LU
x_up_start = x_up_end - LAMBDA_LU
X_UPSTREAM = (x_up_start, x_up_end)

x_down_start = x_end + 0.5 * LAMBDA_LU
x_down_end = x_down_start + LAMBDA_LU
X_DOWNSTREAM = (x_down_start, x_down_end)

# =============================================================================
# 2. FUNKCJE ANALITYCZNE
# =============================================================================
def load_last_period_vtk(folder, prefix, period, y_range):
    search_pattern = os.path.join(folder, f"*{prefix}*.vti")
    files = sorted(glob.glob(search_pattern))
    if not files:
        raise ValueError("Brak plików VTK w katalogu!")
        
    all_iters = []
    for f in files:
        try:
            all_iters.append(int(f.split('_')[-1].split('.')[0]))
        except:
            pass
            
    max_it = max(all_iters)
    min_it = max_it - period
    
    valid_files = [(it, f) for it, f in zip(all_iters, files) if min_it <= it <= max_it]
    time_series, time_array = [], []
    x_coords = None
    
    for it, f in valid_files:
        mesh = pv.read(f)
        rho_flat = mesh['Rho'] if 'Rho' in mesh.array_names else mesh['rho']
            
        is_point_data = (rho_flat.size == np.prod(mesh.dimensions))
        if is_point_data:
            dims = mesh.dimensions
            x_arr = np.arange(dims[0]); y_arr = np.arange(dims[1])
        else:
            dims = (max(1, mesh.dimensions[0]-1), max(1, mesh.dimensions[1]-1), max(1, mesh.dimensions[2]-1))
            x_arr = np.arange(dims[0]) + 0.5; y_arr = np.arange(dims[1]) + 0.5

        if x_coords is None:
            x_coords = x_arr; y_mask = (y_arr >= y_range[0]) & (y_arr <= y_range[1])
            
        rho = rho_flat.reshape(dims, order='F')
        time_series.append(np.mean(rho[:, y_mask, 0], axis=1))
        time_array.append(it)
        
    return x_coords, np.array(time_series), np.array(time_array)

def extract_fourier(time_series, time_array, period_iters):
    omega = 2.0 * np.pi / period_iters
    M = np.column_stack((np.cos(omega * time_array), np.sin(omega * time_array), np.ones_like(time_array)))
    X_sol, _, _, _ = np.linalg.lstsq(M, time_series, rcond=None)
    complex_amp = X_sol[0, :] + 1j * X_sol[1, :]
    return X_sol[2, :], complex_amp, omega

def fit_wave_2way_stable(x, complex_amp, k_guess):
    x0 = x[0] 
    x_loc = x - x0
    
    def residuals(p):
        Af = p[0] + 1j * p[1] 
        Ab = p[2] + 1j * p[3] 
        kr = p[4]; ki = p[5]
        model = Af * np.exp(ki * x_loc) * np.exp(1j * kr * x_loc) + Ab * np.exp(-ki * x_loc) * np.exp(-1j * kr * x_loc)
        diff = model - complex_amp
        return np.concatenate((diff.real, diff.imag))
        
    Af_guess = complex_amp[0]
    p0 = [Af_guess.real, Af_guess.imag, 0.0, 0.0, k_guess, -1e-6]
    lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, 0.5 * k_guess, -0.01]
    upper_bounds = [ np.inf,  np.inf,  np.inf,  np.inf, 1.5 * k_guess,  0.0]
    
    res = least_squares(residuals, p0, method='trf', bounds=(lower_bounds, upper_bounds))
    
    Af_loc = res.x[0] + 1j * res.x[1]
    Ab_loc = res.x[2] + 1j * res.x[3]
    kr = res.x[4]; ki = res.x[5]
    
    return Af_loc, Ab_loc, kr, ki, max(0.0, 1 - (np.sum(np.abs(complex_amp - (Af_loc * np.exp(ki * x_loc) * np.exp(1j * kr * x_loc) + Ab_loc * np.exp(-ki * x_loc) * np.exp(-1j * kr * x_loc)))**2) / np.sum(np.abs(complex_amp - np.mean(complex_amp))**2)))

# =============================================================================
# 3. GŁÓWNA LOGIKA
# =============================================================================
if __name__ == "__main__":
    print(f"--- ROZPOCZĘCIE ANALIZY DLA ABSORBERA (T, R, D) ---")
    print(f"Obliczona długość fali:     {LAMBDA_LU:.1f} lu")
    print(f"Dynamiczne okno dolotowe:   X = {X_UPSTREAM[0]:.1f} do {X_UPSTREAM[1]:.1f}")
    print(f"Lokalizacja strefy badanej: X = {x_start} do {x_end}")
    print(f"Dynamiczne okno wylotowe:   X = {X_DOWNSTREAM[0]:.1f} do {X_DOWNSTREAM[1]:.1f}")
    
    x, t_series, t_array = load_last_period_vtk(args.case_dir, VTK_PREFIX, PERIOD_ITERS, Y_RANGE)
    _, complex_amp, omega = extract_fourier(t_series, t_array, PERIOD_ITERS)
    
    mask_up = (x >= X_UPSTREAM[0]) & (x <= X_UPSTREAM[1])
    mask_down = (x >= X_DOWNSTREAM[0]) & (x <= X_DOWNSTREAM[1])
    
    if not np.any(mask_up) or not np.any(mask_down):
        print("\nBŁĄD: Okna pomiarowe znalazły się poza kanałem!")
        sys.exit(1)
        
    k_guess = omega / C_WAVE
    
    Af_loc_up, Ab_loc_up, kr_up, ki_up, r2_up = fit_wave_2way_stable(x[mask_up], complex_amp[mask_up], k_guess)
    Af_loc_down, Ab_loc_down, kr_down, ki_down, r2_down = fit_wave_2way_stable(x[mask_down], complex_amp[mask_down], k_guess)
    
    # Przeliczenie amplitud bezpośrednio na ścianki absorbera
    dist_in = x_start - x[mask_up][0]
    Af_in = Af_loc_up * np.exp(ki_up * dist_in) * np.exp(1j * kr_up * dist_in)
    Ab_in = Ab_loc_up * np.exp(-ki_up * dist_in) * np.exp(-1j * kr_up * dist_in)
    
    dist_out = x_end - x[mask_down][0]
    Af_out = Af_loc_down * np.exp(ki_down * dist_out) * np.exp(1j * kr_down * dist_out)
    Ab_out = Ab_loc_down * np.exp(-ki_down * dist_out) * np.exp(-1j * kr_down * dist_out)
    
    R_up = abs(Ab_in) / abs(Af_in)
    R_down = abs(Ab_out) / abs(Af_out)
    T_coef = abs(Af_out) / abs(Af_in)
    D_coef = 1.0 - (T_coef**2) - (R_up**2)
    
    print("\n" + "="*50)
    print(" WYNIKI ANALIZY ABSORBERA (Odbicie, Transmisja)")
    print("="*50)
    print(f"Odbicie (R) przed absorberem: {R_up*100:.2f} %")
    print(f"Transmisja (T) za absorberem: {T_coef*100:.2f} %")
    print(f"Dyssypacja (D) w absorberze:  {D_coef*100:.2f} %")
    print(f"Jakość dopasowania (R^2):      {r2_up*100:.2f}% (Dolot) | {r2_down*100:.2f}% (Wylot)")
    print("="*50)
    
    file_exists = os.path.isfile(args.csv)
    with open(args.csv, mode="a", encoding="utf-8") as f:
        if not file_exists or os.stat(args.csv).st_size == 0:
            f.write("period,R_up,T_coef,D_coef,r2_up,r2_down\n")
        f.write(f"{args.period},{R_up:.6f},{T_coef:.6f},{D_coef:.6f},{r2_up:.4f},{r2_down:.4f}\n")