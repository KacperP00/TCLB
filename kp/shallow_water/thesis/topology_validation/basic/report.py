import os
import pandas as pd
import matplotlib.pyplot as plt

# Konfiguracja sciezek wg presetu 'old'
base_dir = "kp/shallow_water/thesis/wyniki_analizy_comparison"

def load_valid_data(filename):
    # Wczytywanie tylko binow zakwalifikowanych jako wiarygodne
    df = pd.read_csv(os.path.join(base_dir, filename))
    return df[df['valid'] == True]

# Pobranie danych z widm
df_euve = load_valid_data("packet_euve_RT_widmo.csv")
df_topo = load_valid_data("packet_topology_RT_widmo.csv")
df_empty = load_valid_data("packet_empty_cav_RT_widmo.csv")

plt.figure(figsize=(12, 7))

# Wykres oparty na wbudowanej zmiennej D_excess (nadmiarowa dyssypacja falochronu)
plt.plot(df_euve['Dlugosc_Fali'], df_euve['D_excess'] * 100, 'b--', linewidth=2, label='Model Euve et al. (Analityczny)')
plt.plot(df_topo['Dlugosc_Fali'], df_topo['D_excess'] * 100, 'g-o', linewidth=2.5, markersize=4, label='Topologia Zoptymalizowana')
plt.plot(df_empty['Dlugosc_Fali'], df_empty['D_excess'] * 100, 'r:^', linewidth=2, markersize=4, label='Puste Wnęki (Brak Topologii)')

plt.axvline(294, color='k', linestyle='--', label='Rezonans Projektowy (294 LU)')

plt.title("Rzeczywista Absorpcja Energii Falochronu (Kompensacja Tłumienia kanału)")
plt.xlabel("Długość fali λ [LU]")
plt.ylabel("Absorpcja Netto (D_excess) [%]")
plt.xlim(225, 350)
plt.ylim(0, 105)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.4)

out_path = os.path.join(base_dir, "MASTER_Wykres_Absorpcji_FIT.png")
plt.savefig(out_path, dpi=300)
print(f"Zapisano zaktualizowany wykres do {out_path}")