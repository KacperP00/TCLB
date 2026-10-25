import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ścieżka do katalogu gdzie diagnostics_03 i singlewave wyrzuciły raporty
base_dir = "kp/shallow_water/thesis/wyniki_analizy_comparison"

# Wczytanie widm z paczki falowej
df_clean = pd.read_csv(os.path.join(base_dir, "packet_clean_canal_2DFFT_widmo.csv"))
df_euve = pd.read_csv(os.path.join(base_dir, "packet_euve_2DFFT_widmo.csv"))
df_topo = pd.read_csv(os.path.join(base_dir, "packet_topology_2DFFT_widmo.csv"))
df_empty = pd.read_csv(os.path.join(base_dir, "packet_empty_cav_2DFFT_widmo.csv"))

lam = df_clean['Dlugosc_Fali']

# Transmisja uśredniona i usunięcie zera (szum)
ct_clean_safe = np.where(df_clean['Ct_Amp'] > 1e-6, df_clean['Ct_Amp'], 1.0)

# Normalizacja - rzeczywista przepuszczalność względem gładkiego kanału
norm_trans_euve = df_euve['Ct_Amp'] / ct_clean_safe
norm_trans_topo = df_topo['Ct_Amp'] / ct_clean_safe
norm_trans_empty = df_empty['Ct_Amp'] / ct_clean_safe

# Rzeczywista absorpcja energetyczna układów: (1 - Transmisja_Znormalizowana^2 - Odbicie^2)
abs_euve = (1.0 - norm_trans_euve**2 - df_euve['Cr_Amp']**2) * 100
abs_topo = (1.0 - norm_trans_topo**2 - df_topo['Cr_Amp']**2) * 100
abs_empty = (1.0 - norm_trans_empty**2 - df_empty['Cr_Amp']**2) * 100

# Wyrzut danych CSV
df_master = pd.DataFrame({
    'Dlugosc_Fali': lam,
    'Absorpcja_Euve_pct': abs_euve,
    'Absorpcja_Topologia_pct': abs_topo,
    'Absorpcja_PusteWneki_pct': abs_empty
})
df_master.to_csv(os.path.join(base_dir, "MASTER_RAPORT_ABSORPCJI.csv"), index=False)

# Rysowanie głównego wykresu do pracy dyplomowej
plt.figure(figsize=(12, 7))

plt.plot(lam, abs_euve, 'b--', linewidth=2, label='Model Euve et al. (Analityczny)')
plt.plot(lam, abs_topo, 'g-', linewidth=2.5, label='Topologia Zoptymalizowana')
plt.plot(lam, abs_empty, 'r:', linewidth=2, label='Puste Wnęki (Brak Topologii)')

plt.axvline(294, color='k', linestyle='--', label='Rezonans Projektowy (294 LU)')

plt.title("Całkowita Rzeczywista Absorpcja Energii (Znormalizowana)")
plt.xlabel("Długość fali λ [LU]")
plt.ylabel("Absorpcja [%]")
plt.xlim(200, 400)
plt.ylim(0, 105)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.4)
plt.savefig(os.path.join(base_dir, "MASTER_Wykres_Absorpcji.png"), dpi=300)
plt.close()

print(f"Raport główny wygenerowany w: {base_dir}")