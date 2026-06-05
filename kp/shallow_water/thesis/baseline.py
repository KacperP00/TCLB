import os
import sys
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv
from scipy.optimize import least_squares

# Konfiguracja wejścia/wyjścia
XML_FILE = "clean_canal.xml"
CASE_DIR = "output/"
CSV_FILE = "baza_fal.csv"

# Wczytanie okresu fali z pliku XML
period_iters = 2828.0
try:
    tree = ET.parse(XML_FILE)
    root = tree.getroot()
    for param in root.findall(".//Param"):
        if param.get("name") == "Wave_Period":
            period_iters = float(param.get("value"))
            break
except Exception:
    print(f"Ostrzeżenie: Nie wczytano {XML_FILE}. Używam {period_iters}")

PERIOD_ITERS = int(np.round(period_iters))
Y_RANGE = (4, 104)

# Zakres przestrzenny pomiaru
X_MEASURE = (400, 2400)

def load_last_period_vtk(folder, prefix, period, y_range):
    # Wczytanie plików VTK z ostatniego okresu fali
    search_pattern = os.path.join(folder, f"*{prefix}*.vti")
    files = sorted(glob.glob(search_pattern))
    if not files:
        raise ValueError("Brak plików VTK w katalogu!")
        
    all_iters = [int(f.split('_')[-1].split('.')[0]) for f in files]
    max_it = max(all_iters)
    min_it = max_it - period
    
    print(f"Analiza okresu: {period} iteracji. Okno czasowe: {min_it} -> {max_it}")
    
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
    # Ekstrakcja amplitudy zespolonej dla zadanej częstotliwości
    omega = 2.0 * np.pi / period_iters
    M = np.column_stack((np.cos(omega * time_array), np.sin(omega * time_array), np.ones_like(time_array)))
    X_sol, _, _, _ = np.linalg.lstsq(M, time_series, rcond=None)
    return X_sol[2, :], X_sol[0, :] + 1j * X_sol[1, :], omega

def fit_wave_2way_stable(x, complex_amp, k_guess):
    # Dopasowanie parametrów fali padającej i odbitej
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
    
    model = Af_loc * np.exp(ki * x_loc) * np.exp(1j * kr * x_loc) + Ab_loc * np.exp(-ki * x_loc) * np.exp(-1j * kr * x_loc)
    ss_res = np.sum(np.abs(complex_amp - model)**2)
    ss_tot = np.sum(np.abs(complex_amp - np.mean(complex_amp))**2)
    r2 = max(0.0, 1 - (ss_res / ss_tot))
    
    # Przesunięcie fazy do początku układu odniesienia (X=0)
    Af_0 = Af_loc * np.exp(-ki * x0) * np.exp(-1j * kr * x0)
    Ab_0 = Ab_loc * np.exp(ki * x0) * np.exp(1j * kr * x0)
    
    return abs(Af_0), np.angle(Af_0), abs(Ab_0), np.angle(Ab_0), kr, ki, r2

if __name__ == "__main__":
    print(f"--- ROZPOCZĘCIE ANALIZY DLA BAZY DANYCH ---")
    
    x, t_series, t_array = load_last_period_vtk(CASE_DIR, "clean_canal_VTK_P00_", PERIOD_ITERS, Y_RANGE)
    _, complex_amp, omega = extract_fourier(t_series, t_array, PERIOD_ITERS)
    
    mask = (x >= X_MEASURE[0]) & (x <= X_MEASURE[1])
    k_guess = omega / 0.1414
    
    A_f, phase_f, A_b, phase_b, kr, ki, r2 = fit_wave_2way_stable(x[mask], complex_amp[mask], k_guess)
    
    print(f"Jakość dopasowania (R^2): {r2*100:.2f}%")
    print("\n[EKSTRAKCJA PARAMETRÓW FALI]")
    print(f"Omega (w):          {omega:.8f}")
    print(f"k_real:             {kr:.8e}")
    print(f"k_imag:             {ki:.8e}")
    print(f"Amplituda padająca: {A_f:.8e}")
    print(f"Faza padająca:      {phase_f:.6f}")
    print(f"Amplituda odbita:   {A_b:.8e}")
    print(f"Faza odbita:        {phase_b:.6f}")
    
    print("\n[BLOK XML DLA FALI PADAJĄCEJ]")
    print(f'<Param name="Wave_A" value="{A_f:.8e}"/>')
    print(f'<Param name="Wave_k_real" value="{kr:.8e}"/>')
    print(f'<Param name="Wave_k_imag" value="{ki:.8e}"/>')
    print(f'<Param name="Wave_w" value="{omega:.8f}"/>')
    print(f'<Param name="Wave_Phase" value="{phase_f:.6f}"/>')
    
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", encoding="utf-8") as f:
        if not file_exists or os.stat(CSV_FILE).st_size == 0:
            f.write("period_iters,omega,k_real,k_imag,A_forward,phase_forward,A_backward,phase_backward,r2_fit\n")
        f.write(f"{PERIOD_ITERS},{omega:.8e},{kr:.8e},{ki:.8e},{A_f:.8e},{phase_f:.6f},{A_b:.8e},{phase_b:.6f},{r2:.4f}\n")
        
    print(f"\n--- ZAPISANO PEŁNE WIDMO PARAMETRÓW DO {CSV_FILE} ---")