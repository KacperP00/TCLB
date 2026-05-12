import os
import sys
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# =============================================================================
# 1. KONFIGURACJA
# =============================================================================
CASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "output/"
VTK_PREFIX = "01clean_canal_VTK_P00_"

GRAVITY = 0.0005
HEIGHT = 40.0
C_WAVE = np.sqrt(GRAVITY * HEIGHT)
LAMBDA_LU = 1200.0
PERIOD_ITERS = int(np.round(LAMBDA_LU / C_WAVE))

MIN_ITER = 59000 
MAX_ITER = MIN_ITER + PERIOD_ITERS 

Y_RANGE = (16, 336)
X_UPSTREAM = (3500, 4700)
X_DOWNSTREAM = (5100, 6300)
X_START = 4800.0
X_END = 6000.0

# =============================================================================
# 2. FUNKCJE ANALITYCZNE
# =============================================================================
def load_vtk_sequence(folder, prefix, min_it, max_it, y_range):
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
        raise ValueError("Nie znaleziono plików VTK!")
        
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

def extract_fourier(time_series, time_array, period_iters):
    omega = 2.0 * np.pi / period_iters
    M = np.column_stack((np.cos(omega * time_array), np.sin(omega * time_array), np.ones_like(time_array)))
    X_sol, _, _, _ = np.linalg.lstsq(M, time_series, rcond=None)
    
    # KLUCZOWA POPRAWKA ZNAKU: Fala kręci się we właściwą stronę
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
        
    Af_guess_c = complex_amp[0]
    p0 = [Af_guess_c.real, Af_guess_c.imag, 0.0, 0.0, k_guess, -1e-6]
    
    # Granice: Tłumienie (ki) nie może być dodatnie, kr w granicach tolerancji
    lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, 0.5 * k_guess, -0.01]
    upper_bounds = [ np.inf,  np.inf,  np.inf,  np.inf, 1.5 * k_guess,  0.0]
    
    res = least_squares(residuals, p0, method='trf', bounds=(lower_bounds, upper_bounds))
    
    Af_loc = res.x[0] + 1j * res.x[1]
    Ab_loc = res.x[2] + 1j * res.x[3]
    kr = res.x[4]; ki = res.x[5]
    
    model = Af_loc * np.exp(ki * x_loc) * np.exp(1j * kr * x_loc) + Ab_loc * np.exp(-ki * x_loc) * np.exp(-1j * kr * x_loc)
    ss_res = np.sum(np.abs(complex_amp - model)**2)
    ss_tot = np.sum(np.abs(complex_amp - np.mean(complex_amp))**2)
    r2 = max(0.0, 1 - (ss_res / ss_tot))
    
    Af_0 = Af_loc * np.exp(-ki * x0) * np.exp(-1j * kr * x0)
    
    return Af_0, Af_loc, Ab_loc, kr, ki, r2, model

# =============================================================================
# 3. GŁÓWNA LOGIKA
# =============================================================================
if __name__ == "__main__":
    print("Rozpoczęcie analizy fali LBM (Stabilny Solver LSTSQ)...")
    
    x, t_series, t_array = load_vtk_sequence(CASE_DIR, VTK_PREFIX, MIN_ITER, MAX_ITER, Y_RANGE)
    _, complex_amp, omega_tu = extract_fourier(t_series, t_array, PERIOD_ITERS)
    
    mask_up = (x >= X_UPSTREAM[0]) & (x <= X_UPSTREAM[1])
    mask_down = (x >= X_DOWNSTREAM[0]) & (x <= X_DOWNSTREAM[1])
    
    k_guess = 2.0 * np.pi / LAMBDA_LU
    
    Af_0_up, Af_loc_up, Ab_loc_up, kr_up, ki_up, r2_up, mod_up = fit_wave_2way_stable(x[mask_up], complex_amp[mask_up], k_guess)
    Af_0_down, Af_loc_down, Ab_loc_down, kr_down, ki_down, r2_down, mod_down = fit_wave_2way_stable(x[mask_down], complex_amp[mask_down], k_guess)
    
    R_up = abs(Ab_loc_up) / abs(Af_loc_up)
    R_down = abs(Ab_loc_down) / abs(Af_loc_down)
    
    dist_in = X_START - x[mask_up][0]
    A_in = Af_loc_up * np.exp(ki_up * dist_in) * np.exp(1j * kr_up * dist_in)
    
    dist_out = X_END - x[mask_down][0]
    A_out = Af_loc_down * np.exp(ki_down * dist_out) * np.exp(1j * kr_down * dist_out)
    
    T_coef = abs(A_out) / abs(A_in)
    D_coef = 1.0 - (T_coef**2) - (R_up**2)
    
    print("\n" + "="*50)
    print(" WYNIKI ANALIZY KINEMATYCZNEJ")
    print("="*50)
    print(f"Odbicie (R) przed wnęką: {R_up*100:.2f} %")
    print(f"Odbicie (R) za wnęką:    {R_down*100:.2f} %")
    print(f"Transmisja (T):          {T_coef*100:.2f} %")
    print(f"Dyssypacja (D):          {D_coef*100:.2f} %")
    print(f"Tłumienie lepkościowe:   {ki_up:.8e} 1/lu")
    print(f"Jakość dopasowania (R^2): {r2_up*100:.2f}% (Dolot) | {r2_down*100:.2f}% (Wylot)")
    
    print("\n[GOTOWY BLOK DO PLIKU 01clean_canal.xml]")
    print(f'        <Param name="Wave_A" value="{abs(Af_0_up):.6f}"/>')
    print(f'        <Param name="Wave_k_real" value="{kr_up:.8f}"/>')
    print(f'        <Param name="Wave_k_imag" value="{ki_up:.8e}"/>')
    print(f'        <Param name="Wave_w" value="{omega_tu:.8f}"/>')
    print(f'        <Param name="Wave_Phase" value="{np.angle(Af_0_up):.6f}"/>')
    print("="*50)