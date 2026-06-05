import os
import sys
import glob
import numpy as np
import pyvista as pv
from scipy.optimize import least_squares

# =============================================================================
# KONFIGURACJA
# =============================================================================
CASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "output/"
VTK_PREFIX = "01clean_canal_VTK_P00_"

# Okno o najniższym błędzie WaveError
MIN_ITER = 30000
MAX_ITER = 47000

GRAVITY = 0.0005
HEIGHT = 40.0
C_WAVE = np.sqrt(GRAVITY * HEIGHT)
LAMBDA_LU = 1200.0
PERIOD_ITERS = int(np.round(LAMBDA_LU / C_WAVE))

# Pomiar na odcinku dolotowym przed wnęką
Y_RANGE = (16, 336)
X_RANGE = (3000, 5000)

# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================
def load_vtk_sequence(folder, prefix, min_it, max_it, y_range):
    # Wczytanie plików VTK z zadanego przedziału iteracji
    search_pattern = os.path.join(folder, f"*{prefix}*.vti")
    files = sorted(glob.glob(search_pattern))
    valid_files = []
    
    for f in files:
        try:
            it = int(f.split('_')[-1].split('.')[0])
            if min_it <= it <= max_it:
                valid_files.append((it, f))
        except ValueError:
            pass
            
    if not valid_files:
        raise ValueError(f"Brak plików VTK w przedziale {min_it}-{max_it}.")
        
    print(f"Wczytano {len(valid_files)} ramek VTK.")
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

def extract_fourier_1way(time_series, time_array, period_iters):
    # Metoda najmniejszych kwadratów do wyznaczenia zespolonej amplitudy
    omega = 2.0 * np.pi / period_iters
    M = np.column_stack((np.cos(omega * time_array), np.sin(omega * time_array), np.ones_like(time_array)))
    X_sol, _, _, _ = np.linalg.lstsq(M, time_series, rcond=None)
    
    # +1j zapewnia zgodność kierunku propagacji w prawo
    complex_amp = X_sol[0, :] + 1j * X_sol[1, :]
    return complex_amp, omega

def fit_single_wave(x, complex_amp, k_guess):
    # Dopasowanie modelu jednokierunkowego (fala padająca z tłumieniem)
    x0 = x[0] 
    x_loc = x - x0
    
    def residuals(p):
        A = p[0] + 1j * p[1] 
        kr = p[2]; ki = p[3]
        model = A * np.exp(ki * x_loc) * np.exp(1j * kr * x_loc)
        diff = model - complex_amp
        return np.concatenate((diff.real, diff.imag))
        
    A_guess_c = complex_amp[0]
    p0 = [A_guess_c.real, A_guess_c.imag, k_guess, -1e-6]
    
    # ki <= 0.0: wymuszenie braku generacji energii
    lower_bounds = [-np.inf, -np.inf, 0.5 * k_guess, -0.01]
    upper_bounds = [ np.inf,  np.inf, 1.5 * k_guess,  0.0]
    
    res = least_squares(residuals, p0, method='trf', bounds=(lower_bounds, upper_bounds))
    
    A_loc = res.x[0] + 1j * res.x[1]
    kr = res.x[2]; ki = res.x[3]
    
    # Powrót do globalnego układu współrzędnych
    A_global = A_loc * np.exp(-ki * x0) * np.exp(-1j * kr * x0)
    
    return A_global, kr, ki

# =============================================================================
# GŁÓWNA LOGIKA
# =============================================================================
if __name__ == "__main__":
    print("Rozpoczęcie ekstrakcji parametrów fali LBM...")
    
    x, t_series, t_array = load_vtk_sequence(CASE_DIR, VTK_PREFIX, MIN_ITER, MAX_ITER, Y_RANGE)
    complex_amp, omega_tu = extract_fourier_1way(t_series, t_array, PERIOD_ITERS)
    
    mask = (x >= X_RANGE[0]) & (x <= X_RANGE[1])
    k_guess = 2.0 * np.pi / LAMBDA_LU
    
    A_global, kr, ki = fit_single_wave(x[mask], complex_amp[mask], k_guess)
    
    print("\n[GOTOWY BLOK DO PLIKU XML]")
    print(f'        <Param name="Wave_A" value="{abs(A_global):.6f}"/>')
    print(f'        <Param name="Wave_k_real" value="{kr:.8f}"/>')
    print(f'        <Param name="Wave_k_imag" value="{ki:.8e}"/>')
    print(f'        <Param name="Wave_w" value="{omega_tu:.8f}"/>')
    print(f'        <Param name="Wave_Phase" value="{np.angle(A_global):.6f}"/>')
    print("="*50)