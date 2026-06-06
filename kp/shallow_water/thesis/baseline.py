import os
import sys
import glob
import numpy as np
import pyvista as pv
from scipy.optimize import least_squares

# Weryfikacja argumentu wejściowego
if len(sys.argv) < 2:
    raise ValueError("Błąd: Wymagany argument <lambda>. Użycie: python3 baseline.py <lambda>")
current_lambda = float(sys.argv[1])

# Obliczenie parametrów fali
GRAVITY = 0.002094
HEIGHT = 9.55
c = np.sqrt(GRAVITY * HEIGHT)
PERIOD_ITERS = int(np.round(current_lambda / c))

# Konfiguracja ścieżek
CASE_DIR = "output/"
CSV_FILE = "kp/shallow_water/thesis/baza_fal.csv"
PREFIX = "clean_canal_VTK_P00_"

Y_RANGE = (4, 104)
X_MEASURE = (400, 2400)

def load_last_period_vtk(folder, prefix, period, y_range):
    # Wyszukiwanie plików VTK, również w podkatalogach
    search_pattern = os.path.join(folder, "**", f"*{prefix}*.vti")
    files = sorted(glob.glob(search_pattern, recursive=True))
    
    if not files:
        raise ValueError(f"Brak plików VTK dla wzorca: {search_pattern}")
        
    all_iters = [int(f.split('_')[-1].split('.')[0]) for f in files]
    max_it = max(all_iters)
    min_it = max_it - period
    
    valid_files = [(it, f) for it, f in zip(all_iters, files) if min_it <= it <= max_it]
    time_series, time_array = [], []
    x_coords = None
    
    for it, f in valid_files:
        mesh = pv.read(f)
        rho_flat = mesh['Rho'] if 'Rho' in mesh.array_names else mesh['rho']
        
        # Poprawne pobranie absolutnych współrzędnych przestrzennych z pliku VTK
        x_bounds = mesh.bounds[0:2] # min_x, max_x
        y_bounds = mesh.bounds[2:4] # min_y, max_y
        
        dims = (mesh.dimensions[0]-1, mesh.dimensions[1]-1) if rho_flat.size < np.prod(mesh.dimensions) else mesh.dimensions[0:2]
        
        x_arr = np.linspace(x_bounds[0], x_bounds[1], dims[0], endpoint=False) + (mesh.spacing[0]/2 if rho_flat.size < np.prod(mesh.dimensions) else 0)
        y_arr = np.linspace(y_bounds[0], y_bounds[1], dims[1], endpoint=False) + (mesh.spacing[1]/2 if rho_flat.size < np.prod(mesh.dimensions) else 0)

        if x_coords is None:
            x_coords = x_arr
            y_mask = (y_arr >= y_range[0]) & (y_arr <= y_range[1])
            
        rho = rho_flat.reshape(dims, order='F')
        # Dynamiczne dopasowanie do wymiarowości zwracanej przez PyVista
        if rho.ndim == 3:
            y_slice = rho[:, y_mask, 0]
        else:
            y_slice = rho[:, y_mask]
        
        time_series.append(np.mean(y_slice, axis=1))
        time_array.append(it)
        
    return x_coords, np.array(time_series), np.array(time_array)

def extract_fourier(time_series, time_array, period_iters):
    # Ekstrakcja składowej podstawowej fali
    omega = 2.0 * np.pi / period_iters
    M = np.column_stack((np.cos(omega * time_array), np.sin(omega * time_array), np.ones_like(time_array)))
    X_sol, _, _, _ = np.linalg.lstsq(M, time_series, rcond=None)
    return X_sol[2, :], X_sol[0, :] + 1j * X_sol[1, :], omega

def fit_wave_2way_stable(x, complex_amp, k_guess):
    # Dopasowanie modelu fali padającej i odbitej
    x0 = x[0] 
    x_loc = x - x0
    
    def residuals(p):
        Af = p[0] + 1j * p[1] 
        Ab = p[2] + 1j * p[3] 
        kr, ki = p[4], p[5]
        model = Af * np.exp(ki * x_loc + 1j * kr * x_loc) + Ab * np.exp(-ki * x_loc - 1j * kr * x_loc)
        diff = model - complex_amp
        return np.concatenate((diff.real, diff.imag))
        
    p0 = [complex_amp[0].real, complex_amp[0].imag, 0.0, 0.0, k_guess, -1e-6]
    lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, 0.5 * k_guess, -0.01]
    upper_bounds = [ np.inf,  np.inf,  np.inf,  np.inf, 1.5 * k_guess,  0.0]
    
    res = least_squares(residuals, p0, method='trf', bounds=(lower_bounds, upper_bounds))
    
    Af_loc = res.x[0] + 1j * res.x[1]
    Ab_loc = res.x[2] + 1j * res.x[3]
    kr, ki = res.x[4], res.x[5]
    
    model = Af_loc * np.exp(ki * x_loc + 1j * kr * x_loc) + Ab_loc * np.exp(-ki * x_loc - 1j * kr * x_loc)
    ss_res = np.sum(np.abs(complex_amp - model)**2)
    ss_tot = np.sum(np.abs(complex_amp - np.mean(complex_amp))**2)
    r2 = max(0.0, 1 - (ss_res / ss_tot))
    
    Af_0 = Af_loc * np.exp(-ki * x0 - 1j * kr * x0)
    Ab_0 = Ab_loc * np.exp(ki * x0 + 1j * kr * x0)
    
    return abs(Af_0), np.angle(Af_0), abs(Ab_0), np.angle(Ab_0), kr, ki, r2

if __name__ == "__main__":
    print(f"--- ANALIZA DLA LAMBDA = {current_lambda} ---")
    
    x, t_series, t_array = load_last_period_vtk(CASE_DIR, PREFIX, PERIOD_ITERS, Y_RANGE)
    _, complex_amp, omega = extract_fourier(t_series, t_array, PERIOD_ITERS)
    
    mask = (x >= X_MEASURE[0]) & (x <= X_MEASURE[1])
    k_guess = omega / 0.1414
    
    A_f, phase_f, A_b, phase_b, kr, ki, r2 = fit_wave_2way_stable(x[mask], complex_amp[mask], k_guess)
    
    # Zapis wyników do pliku CSV
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", encoding="utf-8") as f:
        if not file_exists or os.stat(CSV_FILE).st_size == 0:
            f.write("lambda,period_iters,omega,k_real,k_imag,A_forward,phase_forward,A_backward,phase_backward,r2_fit\n")
        f.write(f"{current_lambda},{PERIOD_ITERS},{omega:.8e},{kr:.8e},{ki:.8e},{A_f:.8e},{phase_f:.6f},{A_b:.8e},{phase_b:.6f},{r2:.4f}\n")
        
    print(f"R^2: {r2*100:.2f}% | Lambda: {current_lambda} | Plik CSV zaktualizowany.")