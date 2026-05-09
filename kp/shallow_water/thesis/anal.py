import numpy as np
import os
import glob
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Konfiguracja ścieżek i parametrów
CASE_DIR = "output/" # Ścieżka do konkretnego wariantu
VTK_PREFIX = "02empty_cav_VTK_P00_"

LAMBDA_LU = 320.0
PERIOD_ITERS = 2263
VTK_STEP = 50 # Zgodnie z nowym krokiem zapisu
MIN_ITER = 11315 # Odcięcie fazy rozbiegu

# Geometria stref pomiarowych
Y_RANGE = (16, 336) # Światło kanału dla uśredniania poprzecznego
X_UPSTREAM = (400, 1000) # Strefa pomiaru przed wnęką
X_DOWNSTREAM = (1800, 2400) # Strefa pomiaru za wnęką

def load_and_average_vtk(folder, prefix, min_iter, y_range):
    # Wyszukiwanie i filtrowanie plików
    search_pattern = os.path.join(folder, f"*{prefix}*.vti")
    files = sorted(glob.glob(search_pattern))
    valid_files = [f for f in files if int(f.split('_')[-1].split('.')[0]) >= min_iter]
    
    if not valid_files:
        raise FileNotFoundError("Brak plików w podanym katalogu.")

    # Pobranie wymiarów domeny 3D (X, Y, Z) z pierwszego pliku
    first_mesh = pv.read(valid_files[0]).cell_data_to_point_data()
    nx = first_mesh.dimensions[0]
    data_matrix = np.zeros((len(valid_files), nx))
    
    # Ekstrakcja danych i uśrednianie
    for t_idx, filepath in enumerate(valid_files):
        mesh = pv.read(filepath).cell_data_to_point_data()
        dims = mesh.dimensions
        
        # Reshape do struktury 3D (Z, Y, X) wymuszonej przez układ pamięci VTK
        rho_3d = mesh.point_data["Rho"].reshape((dims[2], dims[1], dims[0]))
        
        # Ekstrakcja dolnej płaszczyzny 2D (z=0)
        rho_2d = rho_3d[0, :, :]
        
        # Całkowanie sygnału do postaci 1D
        data_matrix[t_idx, :] = np.mean(rho_2d[y_range[0]:y_range[1], :], axis=0)
        
    return data_matrix

def extract_envelope(data_matrix, dt, T_period):
    # Usunięcie składowej stałej (DC offset).
    # Centruje falę na osi Y = 0, eliminując przeciek widma z 0 Hz.
    data_ac = data_matrix - np.mean(data_matrix, axis=0)
    
    # Transformata Fouriera na wyczyszczonym sygnale
    time_steps = data_ac.shape[0]
    fft_data = np.fft.fft(data_ac, axis=0)
    freqs = np.fft.fftfreq(time_steps, d=dt)
    
    # Izolacja częstotliwości fali wymuszającej
    target_f = 1.0 / T_period
    idx = np.argmin(np.abs(freqs[freqs > 0] - target_f)) + 1
    
    # Zwrot zespolonej amplitudy
    return (fft_data[idx, :] * 2.0) / time_steps

def fit_standing_wave(x_coords, complex_amp, k):
    # Definicja funkcji błędu dla optymalizatora
    def residuals(vars):
        A_in = vars[0] + 1j * vars[1]
        A_ref = vars[2] + 1j * vars[3]
        model = A_in * np.exp(-1j * k * x_coords) + A_ref * np.exp(1j * k * x_coords)
        diff = complex_amp - model
        return np.concatenate((np.real(diff), np.imag(diff)))

    # Minimalizacja błędu średniokwadratowego
    res = least_squares(residuals, [0.1, 0.0, 0.0, 0.0])
    return res.x[0] + 1j * res.x[1], res.x[2] + 1j * res.x[3]

def fit_progressive_wave(x_coords, complex_amp, k):
    # Definicja funkcji błędu dla fali transmitowanej
    def residuals(vars):
        A_trans = vars[0] + 1j * vars[1]
        model = A_trans * np.exp(-1j * k * x_coords)
        diff = complex_amp - model
        return np.concatenate((np.real(diff), np.imag(diff)))

    res = least_squares(residuals, [0.1, 0.0])
    return res.x[0] + 1j * res.x[1]

def main():
    print(f"Analiza katalogu: {CASE_DIR}")
    data_matrix = load_and_average_vtk(CASE_DIR, VTK_PREFIX, MIN_ITER, Y_RANGE)
    
    complex_amp = extract_envelope(data_matrix, VTK_STEP, PERIOD_ITERS)
    k = 2 * np.pi / LAMBDA_LU
    x_full = np.arange(0, complex_amp.shape[0])
    
    mask_up = (x_full >= X_UPSTREAM[0]) & (x_full <= X_UPSTREAM[1])
    mask_down = (x_full >= X_DOWNSTREAM[0]) & (x_full <= X_DOWNSTREAM[1])
    
    x_up = x_full[mask_up]
    x_down = x_full[mask_down]
    amp_up = complex_amp[mask_up]
    amp_down = complex_amp[mask_down]
    
    A_in, A_ref = fit_standing_wave(x_up, amp_up, k)
    A_trans = fit_progressive_wave(x_down, amp_down, k)
    
    # Obliczenie modeli do oceny błędów
    model_up = A_in * np.exp(-1j * k * x_up) + A_ref * np.exp(1j * k * x_up)
    model_down = A_trans * np.exp(-1j * k * x_down)
    
    # Obliczenie błędu RMSE
    rmse_up = np.sqrt(np.mean(np.abs(amp_up - model_up)**2))
    rmse_down = np.sqrt(np.mean(np.abs(amp_down - model_down)**2))
    
    R = np.abs(A_ref) / np.abs(A_in)
    T = np.abs(A_trans) / np.abs(A_in)
    D = 1.0 - (R**2 + T**2)
    
    # --- SFORMATOWANY WYDRUK WYNIKÓW ---
    print("\n" + "="*50)
    print("RAPORT Z ANALIZY ZESPOLONEJ")
    print("="*50)
    
    print("\n[1] FALA PADAJĄCA (Inlet)")
    print(f"    Zespolona:  {A_in.real:8.5f} + {A_in.imag:8.5f}j")
    print(f"    Amplituda:  {np.abs(A_in):8.5f} lu")
    print(f"    Faza:       {np.angle(A_in, deg=True):8.2f}°")
    
    print("\n[2] FALA ODBITA (Reflection)")
    print(f"    Zespolona:  {A_ref.real:8.5f} + {A_ref.imag:8.5f}j")
    print(f"    Amplituda:  {np.abs(A_ref):8.5f} lu")
    print(f"    Faza:       {np.angle(A_ref, deg=True):8.2f}°")
    print(f"    RMSE Fit:   {rmse_up:8.6f}")
    
    print("\n[3] FALA TRANSMITOWANA (Transmission)")
    print(f"    Zespolona:  {A_trans.real:8.5f} + {A_trans.imag:8.5f}j")
    print(f"    Amplituda:  {np.abs(A_trans):8.5f} lu")
    print(f"    Faza:       {np.angle(A_trans, deg=True):8.2f}°")
    print(f"    RMSE Fit:   {rmse_down:8.6f}")
    
    print("\n[4] WSPÓŁCZYNNIKI FIZYCZNE")
    print(f"    Współczynnik Odbicia (R):    {R:.4f}")
    print(f"    Współczynnik Transmisji (T): {T:.4f}")
    print(f"    Dyssypacja Energii (D):      {D:.4f}")
    print("="*50 + "\n")

    # --- WIZUALIZACJA I ZAPIS DO PLIKU ---
    plt.figure(figsize=(12, 5))
    plt.plot(x_up, np.abs(amp_up), 'b.', alpha=0.5, label='FFT Amplituda (Dolot)')
    plt.plot(x_down, np.abs(amp_down), 'r.', alpha=0.5, label='FFT Amplituda (Wylot)')
    
    plt.plot(x_up, np.abs(model_up), 'k-', linewidth=2, label='Model Analityczny (Dolot)')
    plt.plot(x_down, np.abs(model_down), 'k-', linewidth=2, label='Model Analityczny (Wylot)')
    
    plt.title(f"Obwiednia fali | R={R:.3f}, T={T:.3f}, D={D:.3f}")
    plt.xlabel("Pozycja X [lu]")
    plt.ylabel("Amplituda [lu]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zapis do pliku zamiast wywoływania interaktywnego okna
    output_img = os.path.join(CASE_DIR, "obwiednia_wyniki.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Wykres obwiedni zapisano pomyślnie jako: {output_img}")

if __name__ == "__main__":
    main()