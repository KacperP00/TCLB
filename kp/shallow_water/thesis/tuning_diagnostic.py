import os
import sys
import glob
import numpy as np
import pyvista as pv
from scipy.optimize import least_squares

# Konfiguracja - dopasuj do swojej symulacji
CASE_DIR = "output/"
VTK_PREFIX = "01clean_canal_VTK_P00_"
PERIOD_ITERS = 2263
MIN_ITER = 22000 
MAX_ITER = 26526

# Obszar diagnostyczny (tam gdzie planujesz Obj2)
X_SCAN = (500, 820) 
Y_CORE = (160, 192)   # Rdzeń fali (środek)
Y_FULL = (16, 336)    # Cała szerokość (ze ścianami)

def load_data(folder, prefix, min_iter, max_iter, x_range):
    files = sorted(glob.glob(os.path.join(folder, f"*{prefix}*.vti")))
    valid_files = [f for f in files if min_iter <= int(f.split('_')[-1].split('.')[0]) <= max_iter]
    
    if not valid_files: raise FileNotFoundError("Brak plików VTK.")

    x_coords = np.arange(x_range[0], x_range[1])
    # Macierz 3D: Czas x Y x X
    raw_data = []
    time_steps = []
    
    print(f"Wczytywanie {len(valid_files)} plików...")
    for filepath in valid_files:
        time_steps.append(int(filepath.split('_')[-1].split('.')[0]))
        mesh = pv.read(filepath).cell_data_to_point_data()
        dims = mesh.dimensions
        rho = mesh.point_data["Rho"].reshape((dims[2], dims[1], dims[0]))[0, :, x_range[0]:x_range[1]]
        raw_data.append(rho)
        
    return np.array(raw_data), np.array(time_steps), x_coords

def analyze_wave(data_3d, time_steps, omega):
    t_window = time_steps[-1] - time_steps[0]
    window = np.hanning(len(time_steps))
    
    # Transformata Fouriera dla każdego punktu (Y, X)
    print("Analiza widmowa pola 2D...")
    complex_field = np.zeros((data_3d.shape[1], data_3d.shape[2]), dtype=complex)
    
    for y in range(data_3d.shape[1]):
        for x in range(data_3d.shape[2]):
            signal = (data_3d[:, y, x] - np.mean(data_3d[:, y, x])) * window
            ft = np.trapz(signal * np.exp(-1j * omega * time_steps), x=time_steps)
            complex_field[y, x] = (ft / t_window) * 4.0 # Korekcja okna
            
    return complex_field

def main():
    omega = 2.0 * np.pi / PERIOD_ITERS
    data_3d, time_steps, x_coords = load_data(CASE_DIR, VTK_PREFIX, MIN_ITER, MAX_ITER, X_SCAN)
    
    # 1. Średni poziom wody (DC Offset)
    h_global = np.mean(data_3d)
    h_core = np.mean(data_3d[:, Y_CORE[0]:Y_CORE[1], :])
    
    # 2. Ekstrakcja pola zespolonego
    c_field = analyze_wave(data_3d, time_steps, omega)
    
    # 3. Analiza "wygięcia" fali (Transverse profile)
    amp_y = np.mean(np.abs(c_field), axis=1)
    phase_y = np.mean(np.angle(c_field), axis=1)
    
    bending_amp = (np.max(amp_y[Y_FULL[0]:Y_FULL[1]]) - np.min(amp_y[Y_FULL[0]:Y_FULL[1]])) / np.mean(amp_y)
    
    # 4. Fitowanie parametrów 1D (na podstawie rdzenia Y_CORE)
    core_signal = np.mean(c_field[Y_CORE[0]:Y_CORE[1], :], axis=0)
    
    def resid(p, x, obs):
        A = p[0] + 1j*p[1]
        k = p[2] + 1j*p[3]
        return (A * np.exp(-1j * k * x) - obs).view(float)

    res = least_squares(resid, [0.1, 0, 2*np.pi/320, 0], args=(x_coords, core_signal))
    A_fit = res.x[0] + 1j*res.x[1]
    k_fit = res.x[2] + 1j*res.x[3]
    
    # RAPORT DIAGNOSTYCZNY
    print("\n" + "="*50)
    print("RAPORT DIAGNOSTYCZNY KANAŁU")
    print("="*50)
    print(f"1. Średni poziom (Height): {h_global:.10f}")
    print(f"2. Niejednorodność poprzeczna (Bending): {bending_amp*100:.2f}%")
    print(f"   (Jeśli > 1%, Adjoint będzie miał problem z szeroką strefą)")
    print(f"3. Amplituda fali (Wave_A): {abs(A_fit):.10f}")
    print(f"4. Liczba falowa (k_real):  {k_fit.real:.10f}")
    print(f"5. Tłumienie (k_imag):      {k_fit.imag:.10f}")
    print(f"6. Faza (Wave_Phase):      {-np.angle(A_fit):.10f}")
    print("="*50)
    print("PARAMETRY DO XML (Skopiuj wszystko):")
    print(f'<Param name="Height" value="{h_global:.10f}"/>')
    print(f'<Param name="Wave_A" value="{abs(A_fit):.10f}"/>')
    print(f'<Param name="Wave_k_real" value="{k_fit.real:.10f}"/>')
    print(f'<Param name="Wave_k_imag" value="{k_fit.imag:.10f}"/>')
    print(f'<Param name="Wave_w" value="{omega:.10f}"/>')
    print(f'<Param name="Wave_Phase" value="{-np.angle(A_fit):.10f}"/>')

if __name__ == "__main__":
    main()