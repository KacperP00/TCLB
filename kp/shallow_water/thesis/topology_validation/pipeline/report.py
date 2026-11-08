import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Argument 1: pojedyncza skala (np. "1.5") lub lista skal po przecinku (np. "1.0,1.5,2.0")
scales_str = sys.argv[1] if len(sys.argv) > 1 else "1.0"
scales = [float(s) for s in scales_str.split(",")]
root_dir = sys.argv[2] if len(sys.argv) > 2 else "kp/shallow_water/thesis"

def res_dir(sc):
    # FIX-B1: bash tworzy wyniki_fit_2x, a Python z float("2") dawalby wyniki_fit_2.0x - sprawdzamy oba zapisy
    for name in (f"{sc:g}", f"{sc}"):
        p = os.path.join(root_dir, f"wyniki_fit_{name}x")
        if os.path.isdir(p):
            return p
    return None

for sc in scales:
    base_dir = res_dir(sc)
    if base_dir is None:
        print(f"UWAGA: brak katalogu wynikow dla skali {sc} - pomijam")
        continue

    df_clean = pd.read_csv(os.path.join(base_dir, "packet_clean_canal_RT_widmo.csv"))
    df_euve = pd.read_csv(os.path.join(base_dir, "packet_euve_RT_widmo.csv"))
    df_topo = pd.read_csv(os.path.join(base_dir, "packet_topology_RT_widmo.csv"))
    df_empty = pd.read_csv(os.path.join(base_dir, "packet_empty_cav_RT_widmo.csv"))

    df_master = pd.DataFrame({
        "Dlugosc_Fali": df_topo["Dlugosc_Fali"],
        "Dlugosc_Fali_1x": df_topo["Dlugosc_Fali_1x"],
        "Absorpcja_Euve_pct": df_euve["D_excess"] * 100.0,
        "Absorpcja_Topologia_pct": df_topo["D_excess"] * 100.0,
        "Absorpcja_PusteWneki_pct": df_empty["D_excess"] * 100.0,
        "Absorpcja_CzystyKanal_pct": df_clean["D_excess"] * 100.0,
    }).dropna()

    df_master.to_csv(os.path.join(base_dir, "MASTER_RAPORT_ABSORPCJI_FIT.csv"), index=False)

    plt.figure(figsize=(12, 7))
    plt.plot(df_master["Dlugosc_Fali"], df_master["Absorpcja_Euve_pct"], "b--s", lw=2, markersize=4, label="Model Euve et al. (Analityczny)")
    plt.plot(df_master["Dlugosc_Fali"], df_master["Absorpcja_Topologia_pct"], "g-o", lw=2.5, markersize=4, label="Topologia Zoptymalizowana")
    plt.plot(df_master["Dlugosc_Fali"], df_master["Absorpcja_PusteWneki_pct"], "r:^", lw=2, markersize=4, label="Puste Wneki (Brak Topologii)")
    plt.axvline(294.0 * sc, color="k", linestyle="--", label=f"Cel Projektowy ({294.0*sc:.0f} LU)")
    plt.title(f"Rzeczywista Absorpcja Energii D_excess (Siatka {sc}x)")
    plt.xlabel(f"Dlugosc fali na siatce {sc}x [LU]")
    plt.ylabel("Absorpcja Netto [%]")
    plt.xlim(220.0 * sc, 355.0 * sc)
    plt.ylim(-2, 105)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.4)
    plt.savefig(os.path.join(base_dir, f"MASTER_Wykres_Absorpcji_{sc}x.png"), dpi=300)
    plt.close()

# Zbiorczy wykres zbieznosci siatki (jesli podano wiecej niz jedna skale)
if len(scales) > 1:
    plt.figure(figsize=(12, 7))
    styles = ["-", "--", ":", "-."]
    for i, sc in enumerate(scales):
        csv_p = os.path.join(res_dir(sc) or "", "MASTER_RAPORT_ABSORPCJI_FIT.csv")
        if os.path.exists(csv_p):
            df_m = pd.read_csv(csv_p)
            st = styles[i % len(styles)]
            plt.plot(df_m["Dlugosc_Fali_1x"], df_m["Absorpcja_Topologia_pct"], f"g{st}", lw=2.2, label=f"Topologia ({sc}x)")
            plt.plot(df_m["Dlugosc_Fali_1x"], df_m["Absorpcja_Euve_pct"], f"b{st}", lw=1.8, label=f"Model Euve ({sc}x)")

    plt.axvline(294.0, color="k", linestyle="--", alpha=0.7, label="Cel Projektowy (294 LU @ 1.0x)")
    plt.title("Analiza Wrazliwosci na Siatke (Mesh Independence Study)")
    plt.xlabel("Ekwiwalentna dlugosc fali na siatce bazowej 1.0x [LU]")
    plt.ylabel("Absorpcja Netto D_excess [%]")
    plt.xlim(220, 355)
    plt.ylim(0, 105)
    plt.legend(loc="lower left", ncol=2)
    plt.grid(True, alpha=0.4)
    plt.savefig(os.path.join(root_dir, "MESH_INDEPENDENCE_COMPARISON.png"), dpi=300)
    plt.close()