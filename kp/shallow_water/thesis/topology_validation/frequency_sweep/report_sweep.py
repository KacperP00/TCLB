import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Pobranie skali z argumentu
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
res_dir = f"kp/shallow_water/thesis/wyniki_sweep_{scale:.1f}x"

if not os.path.exists(res_dir):
    sys.exit(f"Brak katalogu {res_dir}")

# Słownik modeli odpowiadający przedrostkom z run_2.sh
models = {
    "Model Euve et al.": "euve",
    "Topologia Pierwotna": "topology_orig",
    "Topologia Oczyszczona": "topology_clean",
    "Puste Wneki": "empty_cav"
}

styles = {
    "Model Euve et al.": "b--s",
    "Topologia Pierwotna": "g-o",
    "Topologia Oczyszczona": "m-d",
    "Puste Wneki": "r:^"
}

all_data = {}

# Wczytywanie i łączenie plików CSV dla każdego wariantu
for label, prefix in models.items():
    files = glob.glob(f"{res_dir}/sweep_{prefix}_lam*_RT_widmo.csv")
    if not files:
        print(f"Brak plikow dla: {prefix}")
        continue
    
    dfs = [pd.read_csv(f) for f in files]
    df_concat = pd.concat(dfs, ignore_index=True)
    df_concat = df_concat.sort_values("Dlugosc_Fali_1x")
    all_data[label] = df_concat

if not all_data:
    sys.exit("Nie znaleziono zadnych plikow CSV ze sweepu.")

# Generowanie zbiorczego pliku MASTER CSV
base_label = list(all_data.keys())[0]
df_master = all_data[base_label][["Dlugosc_Fali_1x", "Dlugosc_Fali"]].copy()

for label, df in all_data.items():
    col_name = f"Absorpcja_{label.replace(' ', '_')}_pct"
    df_master[col_name] = df["D_excess"].values * 100.0

csv_out = f"{res_dir}/MASTER_SWEEP_ABSORPCJI_{scale:.1f}x.csv"
df_master.to_csv(csv_out, index=False)
print(f"Zapisano dane tabelaryczne: {csv_out}")

# Generowanie wykresu absorpcji
plt.figure(figsize=(12, 7))
for label, df in all_data.items():
    plt.plot(df["Dlugosc_Fali_1x"], df["D_excess"] * 100.0, styles[label], lw=2.5, markersize=6, label=label)

plt.axvline(294.0, color="k", linestyle="--", label="Cel Euve et al. (294 LU)")
plt.title(f"Zestawienie Absorpcji dla zbioru fal harmonicznych (Skala {scale:.1f}x)")
plt.xlabel("Dlugosc fali na siatce 1.0x [LU]")
plt.ylabel("Absorpcja Netto [%]")
plt.xlim(190, 410)
plt.ylim(-2, 105)
plt.legend(loc="lower left")
plt.grid(True, alpha=0.4)

png_out = f"{res_dir}/MASTER_Wykres_Sweep_{scale:.1f}x.png"
plt.savefig(png_out, dpi=300)
plt.close()
print(f"Zapisano wykres: {png_out}")