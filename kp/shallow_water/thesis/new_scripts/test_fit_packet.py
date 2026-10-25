# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.fft import rfft, rfftfreq
from scipy.optimize import least_squares

# Pobranie argumentow: 1: katalog z VTI, 2: nazwa przypadku, 3: mnoznik siatki (1.0 lub 1.5)
input_dir = sys.argv[1] if len(sys.argv) > 1 else "output/"
case_name = sys.argv[2] if len(sys.argv) > 2 else "clean_canal"
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

out_prefix = f"kp/shallow_water/thesis/wyniki_fit_{scale}x/{case_name}_FIT"
os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

# Parametry fizyczne modelu i przeskalowane wspolrzedne stref
g = 0.00209424
h_base = 9.55
c_wave = np.sqrt(g * h_base)

y_range = (int(110 * scale), int(198 * scale))
x_up_range = (int(1500 * scale), int(3500 * scale))
x_down_range = (int(4500 * scale), int(6500 * scale))
x_start = 4000.0 * scale
x_end = 4400.0 * scale
lam_min, lam_max = 220.0 * scale, 340.0 * scale

def load_spacetime_matrix(folder, y_rng):
    # Wyszukanie i posortowanie plikow VTI po numerze iteracji
    files = sorted(glob.glob(os.path.join(folder, "*.vti")),
                   key=lambda f: int(f.split('_')[-1].split('.')[0]))
    if not files:
        raise FileNotFoundError(f"Brak plikow .vti w katalogu: {folder}")
        
    print(f"Wczytywanie {len(files)} plikow VTI z {folder} (skala {scale}x)...")
    time_series, iterations = [], []
    x_coords = None
    
    for f in files:
        iterations.append(int(f.split('_')[-1].split('.')[0]))
        mesh = pv.read(f)
        rho_flat = mesh['Rho'] if 'Rho' in mesh.array_names else mesh['rho']
        is_point_data = (rho_flat.size == np.prod(mesh.dimensions))
        
        if is_point_data:
            dims = mesh.dimensions
            x_arr = np.arange(dims[0], dtype=float)
            y_arr = np.arange(dims[1], dtype=float)
        else:
            dims = (max(1, mesh.dimensions[0]-1), max(1, mesh.dimensions[1]-1), max(1, mesh.dimensions[2]-1))
            x_arr = np.arange(dims[0], dtype=float) + 0.5
            y_arr = np.arange(dims[1], dtype=float) + 0.5
            
        if x_coords is None:
            x_coords = x_arr
            y_mask = (y_arr >= y_rng[0]) & (y_arr <= y_rng[1])
            
        rho = rho_flat.reshape(dims, order='F')
        time_series.append(np.mean(rho[:, y_mask, 0], axis=1) - h_base)
        
    return x_coords, np.array(time_series), np.array(iterations)

def fit_wave_2way(x_arr, complex_amp, k_guess, ki_fixed=None):
    x0 = x_arr[0]
    x_loc = x_arr - x0
    ki_init = -2e-4 / scale if ki_fixed is None else ki_fixed
    
    # Liniowe oszacowanie punktu startowego amplitud Af i Ab
    M_init = np.column_stack([
        np.exp((ki_init + 1j * k_guess) * x_loc),
        np.exp(-(ki_init + 1j * k_guess) * x_loc)
    ])
    ab_init, _, _, _ = np.linalg.lstsq(M_init, complex_amp, rcond=None)
    Af0, Ab0 = ab_init[0], ab_init[1]
    
    if ki_fixed is None:
        def residuals(p):
            Af = p[0] + 1j * p[1]; Ab = p[2] + 1j * p[3]; kr = p[4]; ki = p[5]
            model = Af * np.exp((ki + 1j * kr) * x_loc) + Ab * np.exp(-(ki + 1j * kr) * x_loc)
            diff = model - complex_amp
            return np.concatenate((diff.real, diff.imag))
            
        p0 = [Af0.real, Af0.imag, Ab0.real, Ab0.imag, k_guess, ki_init]
        bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 0.7 * k_guess, -0.005],
                  [ np.inf,  np.inf,  np.inf,  np.inf, 1.3 * k_guess,  0.0])
        res = least_squares(residuals, p0, method='trf', bounds=bounds)
        Af_loc = res.x[0] + 1j * res.x[1]
        Ab_loc = res.x[2] + 1j * res.x[3]
        kr, ki = res.x[4], res.x[5]
    else:
        def residuals(p):
            Af = p[0] + 1j * p[1]; Ab = p[2] + 1j * p[3]; kr = p[4]
            model = Af * np.exp((ki_fixed + 1j * kr) * x_loc) + Ab * np.exp(-(ki_fixed + 1j * kr) * x_loc)
            diff = model - complex_amp
            return np.concatenate((diff.real, diff.imag))
            
        p0 = [Af0.real, Af0.imag, Ab0.real, Ab0.imag, k_guess]
        bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 0.7 * k_guess],
                  [ np.inf,  np.inf,  np.inf,  np.inf, 1.3 * k_guess])
        res = least_squares(residuals, p0, method='trf', bounds=bounds)
        Af_loc = res.x[0] + 1j * res.x[1]
        Ab_loc = res.x[2] + 1j * res.x[3]
        kr, ki = res.x[4], ki_fixed
        
    model = Af_loc * np.exp((ki + 1j * kr) * x_loc) + Ab_loc * np.exp(-(ki + 1j * kr) * x_loc)
    ss_res = np.sum(np.abs(complex_amp - model)**2)
    ss_tot = np.sum(np.abs(complex_amp - np.mean(complex_amp))**2)
    r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-16))
    return Af_loc, Ab_loc, kr, ki, r2

if __name__ == "__main__":
    x, t_series, iters = load_spacetime_matrix(input_dir, y_range)
    dt = np.mean(np.diff(iters))
    Nt = t_series.shape[0]
    
    # Transformata Fouriera w czasie (bez okna Hanna)
    spec_t = np.conj(rfft(t_series, axis=0)) * (2.0 / Nt)
    freqs = rfftfreq(Nt, d=dt)
    
    mask_f = (freqs >= c_wave / lam_max) & (freqs <= c_wave / lam_min)
    valid_indices = np.where(mask_f)[0]
    
    mask_up = (x >= x_up_range[0]) & (x <= x_up_range[1])
    mask_down = (x >= x_down_range[0]) & (x <= x_down_range[1])
    x_up, x_down = x[mask_up], x[mask_down]
    
    results = []
    for idx_f in valid_indices:
        f_val = freqs[idx_f]
        lam = c_wave / f_val
        k_guess = 2.0 * np.pi / lam
        amp_x = spec_t[idx_f, :]
        
        Af_up, Ab_up, kr_up, ki_up, r2_up = fit_wave_2way(x_up, amp_x[mask_up], k_guess)
        Af_down, Ab_down, kr_down, _, r2_down = fit_wave_2way(x_down, amp_x[mask_down], k_guess, ki_fixed=ki_up)
        
        # Ekstrapolacja do krawedzi wneki X_START i X_END
        dist_in = x_start - x_up[0]
        A_inc_in = Af_up * np.exp((ki_up + 1j * kr_up) * dist_in)
        A_ref_in = Ab_up * np.exp(-(ki_up + 1j * kr_up) * dist_in)
        
        dist_out = x_end - x_down[0]
        A_trans_out = Af_down * np.exp((ki_up + 1j * kr_down) * dist_out)
        
        Cr = abs(A_ref_in) / abs(A_inc_in)
        Ct_edges = abs(A_trans_out) / abs(A_inc_in)
        
        # Kompensacja tlumienia lepkosciowego na samej dlugosci wneki (X_END - X_START)
        Ct_comp = Ct_edges * np.exp(-ki_up * (x_end - x_start))
        Abs_comp_pct = (1.0 - Cr**2 - Ct_comp**2) * 100.0
        
        results.append({
            'Dlugosc_Fali': lam,
            'Cr_pct': Cr * 100.0,
            'Ct_Krawedzie_pct': Ct_edges * 100.0,
            'Ct_Kompensacja_pct': Ct_comp * 100.0,
            'Absorpcja_Netto_pct': Abs_comp_pct,
            'ki_up': ki_up,
            'R2_up': r2_up,
            'R2_down': r2_down
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{out_prefix}_widmo.csv", index=False)
    print(f"\n=== WYNIKI FITTINGU SZEROKOPASMOWEGO ({case_name}, skala {scale}x) ===")
    print(df_res.to_string(index=False))
    print(f"\nSrednia transmisja (Krawedzie X_START->X_END): {df_res['Ct_Krawedzie_pct'].mean():.2f}%")
    print(f"Srednia transmisja (Pelna kompensacja ki):     {df_res['Ct_Kompensacja_pct'].mean():.2f}%")
    print(f"Srednie odbicie Cr:                            {df_res['Cr_pct'].mean():.2f}%")