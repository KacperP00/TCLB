import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Pobranie skali siatki z argumentu wywolania
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
res_dir = f"kp/shallow_water/thesis/wyniki_sweep_{scale:g}x"

if not os.path.exists(res_dir):
    sys.exit(f"Brak katalogu: {res_dir}")

# Mapowanie nazw modeli na przedrostki plikow wygenerowanych w sweepie
models = {
    "Topologia pierwotna": ("topology_orig", "g-o"),
    "Topologia oczyszczona": ("topology_clean", "m-d"),
    "Model Euve et al.": ("euve", "b--s"),
    "Puste wneki": ("empty_cav", "r:^"),
}

data = {}
for label, (prefix, _) in models.items():
    files = glob.glob(f"{res_dir}/sweep_{prefix}_lam*_RT_widmo.csv")
    if not files:
        continue
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    data[label] = df.sort_values("Dlugosc_Fali_1x").reset_index(drop=True)

if not data:
    sys.exit("Brak plikow CSV do analizy.")

# Budowa zbiorczej tabeli z wartosciami D, R oraz T dla kazdego modelu
base_df = next(iter(data.values()))
master = pd.DataFrame({"Dlugosc_Fali_1x": base_df["Dlugosc_Fali_1x"]})

for label, df in data.items():
    tag = label.replace(" ", "_").replace(".", "")
    master[f"D_{tag}_pct"] = df["D_excess"].values * 100.0
    master[f"R_{tag}_pct"] = df["R"].values * 100.0
    master[f"T_{tag}_pct"] = df["T_comp"].values * 100.0

csv_out = f"{res_dir}/MASTER_SWEEP_RT_{scale:g}x.csv"
master.to_csv(csv_out, index=False)
print(f"Zapisano pelna tabele D/R/T: {csv_out}")

# Generowanie 3-panelowego wykresu: Absorpcja D, Odbicie R, Transmisja T
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
metrics = [
    ("D_excess", "Absorpcja D [%]\n(D = 1 − R² − T²)"),
    ("R", "Odbicie R [%]"),
    ("T_comp", "Transmisja T [%]"),
]

for ax, (col, ylabel) in zip(axes, metrics):
    for label, (_, style) in models.items():
        if label in data:
            ax.plot(data[label]["Dlugosc_Fali_1x"], data[label][col] * 100.0,
                    style, lw=2.2, markersize=5.5, label=label)
    ax.axvline(294.0, color="k", linestyle="--", lw=1.1, alpha=0.7, label="Cel projektowy (294 LU)")
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.35)

axes[0].set_title(f"Zestawienie absorpcji (D), odbicia (R) i transmisji (T) – siatka {scale:g}x",
                  fontsize=12, fontweight="bold")
axes[-1].set_xlabel("Dlugosc fali λ [LU]", fontsize=11)
axes[-1].set_xticks(base_df["Dlugosc_Fali_1x"])
axes[-1].set_xlim(190, 410)
axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9.5)

plt.tight_layout()
png_out = f"{res_dir}/MASTER_Wykres_Sweep_DRT_{scale:g}x.png"
plt.savefig(png_out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Zapisano wykres 3-panelowy: {png_out}")