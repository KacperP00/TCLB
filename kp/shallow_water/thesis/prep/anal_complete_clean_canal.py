import os
import sys
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# =============================================================================
# 1. KONFIGURACJA PARAMETRÓW (Zsynchronizowane z wave_inlet_02.py)
# =============================================================================
CASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "output/"
VTK_PREFIX = "01clean_canal_VTK_P00_"

# Fizyka 
GRAVITY = 0.0005
HEIGHT = 40.0
C_WAVE = np.sqrt(GRAVITY * HEIGHT)     # ~0.14142 lu/iter
LAMBDA_LU = 1200.0
PERIOD_ITERS = int(np.round(LAMBDA_LU / C_WAVE)) # 8485 iteracji

# Pomiary czasowe
VTK_STEP = 50   # WAŻNE: W XML musi być <VTK Iterations="50" .../>
MIN_ITER = 25000 # Czekamy aż fala przepłynie przez wnękę i strefę transmisji
MAX_ITER = MIN_ITER + PERIOD_ITERS # Pobieramy dokładnie jeden pełny okres (do 98485)

# Pomiary przestrzenne (Pasy pomiarowe)
Y_RANGE = (16, 336)
X_UPSTREAM = (3000, 5000)    # Czysta strefa przed wnęką
X_DOWNSTREAM = (6500, 8000) # Czysta strefa za wnęką

# Położenie wnęki
X_START = 7200.0
X_END = 8480.0

# =============================================================================
# 2. FUNKCJE POMOCNICZE
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
        except:
            pass
            
    if not valid_files:
        raise ValueError(f"Nie znaleziono plików VTK w przedziale {min_it} - {max_it} iteracji!")
        
    print(f"Wczytano {len(valid_files)} ramek VTK.")
    
    time_series = []
    x_coords = None
    
    for it, f in valid_files:
        mesh = pv.read(f)
        
        # Pobranie plaskiej tablicy danych
        if 'Rho' in mesh.array_names:
            rho_flat = mesh['Rho']
        elif 'rho' in mesh.array_names:
            rho_flat = mesh['rho']
        else:
            raise KeyError(f"CRITICAL: Brak gęstości. Dostępne zmienne: {mesh.array_names}")
            
        # Logika zmiany ksztaltu dla Cell Data lub Point Data
        is_point_data = (rho_flat.size == np.prod(mesh.dimensions))
        
        if is_point_data:
            dims = mesh.dimensions
            x_arr = np.arange(dims[0])
            y_arr = np.arange(dims[1])
        else:
            dims = (max(1, mesh.dimensions[0] - 1), 
                    max(1, mesh.dimensions[1] - 1), 
                    max(1, mesh.dimensions[2] - 1))
            # Wyznaczenie idealnych srodkow komorek (bez dotykania mesh.x/mesh.y)
            x_arr = np.arange(dims[0]) + 0.5
            y_arr = np.arange(dims[1]) + 0.5

        # Inicjalizacja osi w pierwszej iteracji
        if x_coords is None:
            print(f"-> Zmienne VTK: {mesh.array_names}")
            print(f"-> Typ zapisu: {'Point Data' if is_point_data else 'Cell Data'}")
            print(f"-> Skalibrowane wymiary: {dims}")
            x_coords = x_arr
            y_coords = y_arr
            y_mask = (y_coords >= y_range[0]) & (y_coords <= y_range[1])
            
        # Zmiana ksztaltu i redukcja osi
        rho = rho_flat.reshape(dims, order='F')
        
        # Średnia po wysokości kanału (redukcja szumu)
        rho_avg = np.mean(rho[:, y_mask, 0], axis=1)
        time_series.append(rho_avg)
        
    return x_coords, np.array(time_series)

def extract_fourier(time_series, vtk_step, period_iters):
    T = time_series.shape[0]
    mean_height = np.mean(time_series, axis=0)
    fluctuation = time_series - mean_height
    
    time_array = np.arange(T) * vtk_step
    omega = 2.0 * np.pi / period_iters
    
    complex_amp = np.zeros(fluctuation.shape[1], dtype=complex)
    for i in range(fluctuation.shape[1]):
        # Ręczna transformata Fouriera dla częstotliwości podstawowej fali
        z = fluctuation[:, i] * np.exp(1j * omega * time_array)
        complex_amp[i] = 2.0 * np.mean(z)
        
    return mean_height, complex_amp, omega

def fit_wave_spatial(x, complex_amp):
    # Dopasowanie numeryczne: A_c * exp(k_imag * x) * exp(i * k_real * x)
    def residuals(p):
        A_r, A_i, k_r, k_i = p
        A_c = A_r + 1j * A_i
        model = A_c * np.exp(k_i * x) * np.exp(1j * k_r * x)
        diff = model - complex_amp
        return np.concatenate((diff.real, diff.imag))
        
    # Punkty startowe optymalizacji
    k_r_guess = 2.0 * np.pi / LAMBDA_LU
    p0 = [np.mean(np.abs(complex_amp)), 0.0, k_r_guess, 0.0]
    
    res = least_squares(residuals, p0, method='lm')
    A_r, A_i, k_r, k_i = res.x
    return (A_r + 1j * A_i), k_r, k_i

# =============================================================================
# 3. GŁÓWNA LOGIKA SKRYPTU
# =============================================================================
if __name__ == "__main__":
    print("Rozpoczęcie analizy fali LBM...")
    
    x, t_series = load_vtk_sequence(CASE_DIR, VTK_PREFIX, MIN_ITER, MAX_ITER, Y_RANGE)
    
    # 1. Analiza Fouriera (Czasowa)
    print("Wykonywanie transformaty czasowej...")
    _, complex_amp, omega_tu = extract_fourier(t_series, VTK_STEP, PERIOD_ITERS)
    
    # 2. Wycinanie danych dla stref (Przestrzenna)
    mask_up = (x >= X_UPSTREAM[0]) & (x <= X_UPSTREAM[1])
    mask_down = (x >= X_DOWNSTREAM[0]) & (x <= X_DOWNSTREAM[1])
    
    # 3. Dopasowanie parametrów fali
    print("Dopasowywanie obwiedni przestrzennej (Tłumienie i dyspersja)...")
    A_up, kr_up, ki_up = fit_wave_spatial(x[mask_up], complex_amp[mask_up])
    A_down, kr_down, ki_down = fit_wave_spatial(x[mask_down], complex_amp[mask_down])
    
    # Przeliczenie amplitud z uwzględnieniem tłumienia do płaszczyzn wnęki
    A_in = A_up * np.exp(ki_up * X_START) * np.exp(1j * kr_up * X_START)
    A_out = A_down * np.exp(ki_down * X_END) * np.exp(1j * kr_down * X_END)
    
    # Ostateczne wyliczenie współczynnika transmisji i dyssypacji
    T_coef = abs(A_out) / abs(A_in)
    D_coef = 1.0 - (T_coef**2) # Kanał czysty: Odbicie = 0, więc D = 1 - T^2
    
    print("\n" + "="*50)
    print(f" WYNIKI ANALIZY FALI (Czysty Kanał)")
    print("="*50)
    print(f"Transmisja (T):   {T_coef:.6f}")
    print(f"Dyssypacja (D):   {D_coef:.6f}  <- Błąd numeryczny lepkości LBM")
    print(f"Tłumienie lepk.: {ki_up:.8e} 1/lu")
    
    print("\n[GOTOWY BLOK DO PLIKU 01clean_canal.xml]")
    print(f'        <Param name="Wave_A" value="{abs(A_up):.6f}"/>')
    print(f'        <Param name="Wave_k_real" value="{kr_up:.8f}"/>')
    print(f'        <Param name="Wave_k_imag" value="{ki_up:.8e}"/>')
    print(f'        <Param name="Wave_w" value="{omega_tu:.8f}"/>')
    print(f'        <Param name="Wave_Phase" value="{np.angle(A_up):.6f}"/>')
    print("="*50)

    # Opcjonalny wykres dopasowania dla kontroli
    plt.figure(figsize=(10, 5))
    plt.plot(x, np.abs(complex_amp), 'k.', markersize=2, label='Symulacja LBM (Amplituda w węzłach)', alpha=0.3)
    plt.plot(x[mask_up], np.abs(A_up * np.exp(ki_up * x[mask_up])), 'r-', linewidth=2, label='Dopasowanie przed wnęką')
    plt.plot(x[mask_down], np.abs(A_down * np.exp(ki_down * x[mask_down])), 'b-', linewidth=2, label='Dopasowanie za wnęką')
    plt.axvline(X_START, color='g', linestyle='--', label='Początek wnęki')
    plt.axvline(X_END, color='g', linestyle='-.', label='Koniec wnęki')
    plt.xlabel('Oś X [lu]')
    plt.ylabel('Amplituda Fali [lu]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dopasowanie_fali_czysty_kanal.png', dpi=150)
    print("\nZapisano wykres kontrolny 'dopasowanie_fali_czysty_kanal.png'")