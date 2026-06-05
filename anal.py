import numpy as np
import os
import glob
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Konfiguracja ścieżek i parametrów
CASE_DIR = "TCLB/output02" # Ścieżka do konkretnego wariantu
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

    # Pobranie wymiarów domeny z pierwszego pliku po konwersji
    first_mesh = pv.read(valid_files[0]).cell_data_to_point_data()
    nx = first_mesh.dimensions[0]
    data_matrix = np.zeros((len(valid_files), nx))
    
    # Ekstrakcja danych i uśrednianie poprzeczne
    for t_idx, filepath in enumerate(valid_files):
        mesh = pv.read(filepath)
        
        # Konwersja danych z komórek na węzły siatki
        mesh = mesh.cell_data_to_point_data()
        
        dims = mesh.dimensions
        rho = mesh.point_data["Rho"].reshape((dims[1], dims[0]))
        
        # Całkowanie sygnału do postaci 1D
        data_matrix[t_idx, :] = np.mean(rho[y_range[0]:y_range[1], :], axis=0)
        
    return data_matrix

def extract_envelope(data_matrix, dt, T_period):
    # Transformata FFT w dziedzinie czasu
    time_steps = data_matrix.shape[0]
    fft_data = np.fft.fft(data_matrix, axis=0)
    freqs = np.fft.fftfreq(time_steps, d=dt)
    
    # Izolacja częstotliwości podstawowej
    target_f = 1.0 / T_period
    idx = np.argmin(np.abs(freqs[freqs > 0] - target_f)) + 1
    
    # Zespolona amplituda przestrzenna
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
    # Pobranie danych dla wskazanego przypadku
    print(f"Analiza katalogu: {CASE_DIR}")
    data_matrix = load_and_average_vtk(CASE_DIR, VTK_PREFIX, MIN_ITER, Y_RANGE)
    
    # Przetwarzanie sygnału
    complex_amp = extract_envelope(data_matrix, VTK_STEP, PERIOD_ITERS)
    k = 2 * np.pi / LAMBDA_LU
    x_full = np.arange(0, complex_amp.shape[0])
    
    # Separacja stref
    mask_up = (x_full >= X_UPSTREAM[0]) & (x_full <= X_UPSTREAM[1])
    mask_down = (x_full >= X_DOWNSTREAM[0]) & (x_full <= X_DOWNSTREAM[1])
    
    # Dopasowanie modeli z R.py
    A_in, A_ref = fit_standing_wave(x_full[mask_up], complex_amp[mask_up], k)
    A_trans = fit_progressive_wave(x_full[mask_down], complex_amp[mask_down], k)
    
    # Obliczenie współczynników i dyssypacji
    R = np.abs(A_ref) / np.abs(A_in)
    T = np.abs(A_trans) / np.abs(A_in)
    D = 1.0 - (R**2 + T**2)
    
    # Wydruk wyników
    print("-" * 30)
    print(f"Współczynnik odbicia (R):    {R:.4f}")
    print(f"Współczynnik transmisji (T): {T:.4f}")
    print(f"Dyssypacja energii (D):      {D:.4f}")
    print("-" * 30)

    # Wizualizacja obwiedni dla kontroli błędów
    plt.figure(figsize=(12, 5))
    plt.plot(x_full[mask_up], np.abs(complex_amp[mask_up]), 'b.', label='Dane (Dolot)')
    plt.plot(x_full[mask_down], np.abs(complex_amp[mask_down]), 'r.', label='Dane (Wylot)')
    
    model_up = np.abs(A_in * np.exp(-1j * k * x_full[mask_up]) + A_ref * np.exp(1j * k * x_full[mask_up]))
    model_down = np.abs(A_trans * np.exp(-1j * k * x_full[mask_down]))
    
    plt.plot(x_full[mask_up], model_up, 'k-', label='Model analityczny')
    plt.plot(x_full[mask_down], model_down, 'k-')
    
    plt.title(f"Obwiednia fali | R={R:.3f}, T={T:.3f}, D={D:.3f}")
    plt.xlabel("X [lu]")
    plt.ylabel("Amplituda [lu]")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()