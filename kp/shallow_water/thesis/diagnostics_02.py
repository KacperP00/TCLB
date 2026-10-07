import pyvista as pv
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# Konfiguracja środowiska
base_out_dir = "kp/shallow_water/thesis/wyniki_analizy_comparison"
os.makedirs(base_out_dir, exist_ok=True)
report_file = os.path.join(base_out_dir, "raport_zbiorczy.txt")

# Definicja przypadków (czysty kanał musi być pierwszy do kalibracji)
cases = ["clean_canal", "euve_basic", "topology_basic"]

# Parametry fizyczne i numeryczne
y_center = 154
x_probe_up = 2500      
x_probe_down = 4800    
h_base = 9.55
c_wave = np.sqrt(0.00209424 * h_base)
split_iter = 43000     

# Słowniki na dane kalibracyjne i wyniki do wykresów zbiorczych
baseline_amp_trans = None
plot_data = {}

# Inicjalizacja pliku raportu
with open(report_file, "w", encoding="utf-8") as f:
    f.write("ZBIORCZY RAPORT Z ANALIZY WIDMOWEJ (LBM SWE)\n")
    f.write("============================================\n\n")

for case in cases:
    input_dir = f"{case}_output"
    out_prefix = os.path.join(base_out_dir, case)
    
    files = sorted(glob.glob(os.path.join(input_dir, "*.vti")))
    if not files:
        files = sorted(glob.glob(os.path.join(input_dir, "*.pvti")))
        
    if not files:
        print(f"Brak plików dla przypadku: {case}. Pomijam.")
        continue

    print(f"Przetwarzanie przypadku: {case} ({len(files)} plików)")

    iterations = []
    eta_up = []
    eta_down = []

    # Ekstrakcja danych z VTK
    for f_name in files:
        iter_num = int(f_name.split('_')[-1].split('.')[0])
        iterations.append(iter_num)
        
        mesh = pv.read(f_name)
        data_type = "cell_data" if "Rho" in mesh.cell_data else "point_data"
        
        if data_type == "cell_data":
            x_coords = mesh.cell_centers().points[:, 0]
            y_coords = mesh.cell_centers().points[:, 1]
            rho = mesh.cell_data["Rho"]
        else:
            x_coords = mesh.points[:, 0]
            y_coords = mesh.points[:, 1]
            rho = mesh.point_data["Rho"]
        
        dist_up = np.sqrt((x_coords - x_probe_up)**2 + (y_coords - y_center)**2)
        dist_down = np.sqrt((x_coords - x_probe_down)**2 + (y_coords - y_center)**2)
        
        eta_up.append(rho[np.argmin(dist_up)] - h_base)
        eta_down.append(rho[np.argmin(dist_down)] - h_base)

    t = np.array(iterations)
    eta_up = np.array(eta_up)
    eta_down = np.array(eta_down)
    dt = np.mean(np.diff(t))

    # Separacja sygnału w czasie
    eta_inc = np.copy(eta_up)
    eta_ref = np.copy(eta_up)
    mask_ref = t > split_iter
    eta_inc[mask_ref] = 0.0
    eta_ref[~mask_ref] = 0.0

    # FFT z Zero-Padding (poprawa rozdzielczości widmowej)
    N = len(t)
    N_pad = N * 4  
    
    # Wyznaczenie transformat zespolonych
    yf_inc_cplx = rfft(eta_inc, n=N_pad) * (2.0 / N)
    yf_ref_cplx = rfft(eta_ref, n=N_pad) * (2.0 / N)
    yf_trans_cplx = rfft(eta_down, n=N_pad) * (2.0 / N)
    xf = rfftfreq(N_pad, d=dt)

    mask_freq = (xf > 0)
    f_freq = xf[mask_freq]
    
    # Amplitudy
    amp_inc = np.abs(yf_inc_cplx)[mask_freq]
    amp_ref = np.abs(yf_ref_cplx)[mask_freq]
    amp_trans = np.abs(yf_trans_cplx)[mask_freq]

    # Faza odbicia (względem sygnału padającego)
    phase_ref = np.angle(yf_ref_cplx[mask_freq] / yf_inc_cplx[mask_freq])

    valid = amp_inc > (np.max(amp_inc) * 0.05)
    
    # Pierwotne współczynniki amplitudowe i energetyczne
    Cr_amp = np.zeros_like(amp_inc)
    Ct_amp = np.zeros_like(amp_inc)
    Cr_amp[valid] = amp_ref[valid] / amp_inc[valid]
    Ct_amp[valid] = amp_trans[valid] / amp_inc[valid]
    
    Er_raw = Cr_amp**2
    Et_raw = Ct_amp**2

    # Logika normalizacji
    is_baseline = (case == "clean_canal")
    if is_baseline:
        baseline_amp_trans = np.copy(Ct_amp)
        Ct_norm = np.copy(Ct_amp)
    else:
        ct_clean_safe = np.where(baseline_amp_trans > 1e-6, baseline_amp_trans, 1.0)
        Ct_norm = np.zeros_like(amp_inc)
        Ct_norm[valid] = Ct_amp[valid] / ct_clean_safe[valid]
    
    # Znormalizowane wskaźniki energetyczne
    Et_norm = Ct_norm**2
    E_abs_norm = np.zeros_like(amp_inc)
    E_abs_norm[valid] = 1.0 - Er_raw[valid] - Et_norm[valid]

    lambda_wave = c_wave / f_freq

    # Zapis surowych i znormalizowanych danych
    df_spectrum = pd.DataFrame({
        'Czestotliwosc': f_freq,
        'Dlugosc_Fali': lambda_wave,
        'Amp_Inc': amp_inc,
        'Amp_Ref': amp_ref,
        'Amp_Trans': amp_trans,
        'Faza_Odbicia_rad': phase_ref,
        'Cr_Amp_Pierwotne': Cr_amp,
        'Ct_Amp_Pierwotne': Ct_amp,
        'Ct_Amp_Znorm': Ct_norm,
        'Er_Energia': Er_raw,
        'Et_Energia_Znorm': Et_norm,
        'Abs_Energia_Znorm': E_abs_norm
    })
    df_spectrum.to_csv(f"{out_prefix}.csv", index=False)

    # Agregacja danych do wykresu zbiorczego
    plot_data[case] = {
        'lambda': lambda_wave,
        'Cr_amp': Cr_amp,
        'Ct_norm': Ct_norm,
        'valid': valid
    }

    # Statystyki do raportu (dla pasma 200-400 LU)
    band_strict = (lambda_wave >= 200) & (lambda_wave <= 400) & valid
    m_cr = np.mean(Cr_amp[band_strict]) * 100
    m_ct_raw = np.mean(Ct_amp[band_strict]) * 100
    m_ct_norm = np.mean(Ct_norm[band_strict]) * 100
    m_abs_norm = np.mean(E_abs_norm[band_strict]) * 100

    report_text = f"--- PRZYPADEK: {case} ---\n"
    report_text += f"Średnie Odbicie Amplitudowe (C_R): {m_cr:.2f}%\n"
    report_text += f"Średnia Transmisja Pierwotna (C_T_raw): {m_ct_raw:.2f}%\n"
    if not is_baseline:
        report_text += f"Średnia Transmisja Znormalizowana (C_T_norm): {m_ct_norm:.2f}%\n"
        report_text += f"Rzeczywista Absorpcja Energii (E_abs): {m_abs_norm:.2f}%\n"
    report_text += "\n"

    with open(report_file, "a", encoding="utf-8") as f:
        f.write(report_text)

# Generowanie zbiorczego wykresu skuteczności (porównanie struktur)
if "euve_basic" in plot_data and "topology_basic" in plot_data:
    plt.figure(figsize=(12, 7))
    
    # Euve
    lam_e = plot_data['euve_basic']['lambda']
    band_e = (lam_e >= 150) & (lam_e <= 450)
    plt.plot(lam_e[band_e], plot_data['euve_basic']['Ct_norm'][band_e] * 100, 'b--', label='Euve - Transmisja', linewidth=2)
    plt.plot(lam_e[band_e], plot_data['euve_basic']['Cr_amp'][band_e] * 100, 'r--', label='Euve - Odbicie', linewidth=1.5)
    
    # Topologia
    lam_t = plot_data['topology_basic']['lambda']
    band_t = (lam_t >= 150) & (lam_t <= 450)
    plt.plot(lam_t[band_t], plot_data['topology_basic']['Ct_norm'][band_t] * 100, 'g-', label='Topologia - Transmisja', linewidth=2)
    plt.plot(lam_t[band_t], plot_data['topology_basic']['Cr_amp'][band_t] * 100, 'm-', label='Topologia - Odbicie', linewidth=1.5)

    plt.axvspan(200, 400, color='gray', alpha=0.15, label='Pasmo wymuszenia (200-400 LU)')
    plt.axvline(302, color='black', linestyle=':', label='Rezonans punktowy (~302 LU)')
    
    plt.xlim(150, 450)
    plt.ylim(0, 100)
    plt.title("Znormalizowana Skuteczność Widmowa - Porównanie Struktur")
    plt.xlabel("Długość fali λ [LU]")
    plt.ylabel("Współczynnik [%]")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_out_dir, "zbiorcze_widmo_porownawcze.png"), dpi=300)
    plt.close()

print(f"Zakończono. Raport i pliki CSV zapisano w: {base_out_dir}")