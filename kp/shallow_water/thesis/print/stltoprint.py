import os
import sys
import numpy as np
import pyvista as pv

# Parametry wejsciowe: L [mm] oraz wspolczynniki wymiarowe wzgledem L
L_MM = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
ALPHA_HZ = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15     # wysokosc scianek Z = 0.15 * L (30 mm)
ALPHA_BASE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02   # grubosc dna = 0.04 * L (4 mm)
ALPHA_WALL = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02   # grubosc scianek zewn. = 0.04 * L (4 mm)
ADD_CASSETTE = True                                              # True: dodaje dno i 3 scianki zewnetrzne

# stale geometryczne siatki bazowej 1.0x w jednostkach LU
L_LU = 100.0
X_MIN_LU, X_MAX_LU = 4000.0, 4400.0
Y_SPLIT_LU = 154.0
Y_BOT_MIN_LU, Y_BOT_MAX_LU = 4.0, 104.0
Y_TOP_MIN_LU, Y_TOP_MAX_LU = 204.0, 304.0

# Przeliczenie wymiarow fizycznych w milimetrach
scale_xy = L_MM / L_LU
cav_lx = 4.0 * L_MM
cav_ly = 1.0 * L_MM
hz = ALPHA_HZ * L_MM
t_base = ALPHA_BASE * L_MM
t_wall = ALPHA_WALL * L_MM
z_embed = min(0.5, 0.25 * t_base)  # wpuszczenie wysp w plyte denna dla ciaglosci siatki


def build_cassette_shell(lx, ly, hz_val, tb, tw, open_side):
    # Generowanie plyty dennej (z od -tb do 0) oraz 3 scianek zewnetrznych (z od -tb do hz_val)
    parts = []
    # Plyta denna obejmujaca obszar roboczy oraz szerokosc scianek zewnetrznych
    y_min_ext = -tw if open_side == "ymax" else 0.0
    y_max_ext = ly if open_side == "ymax" else ly + tw
    base_box = pv.Box(bounds=(-tw, lx + tw, y_min_ext, y_max_ext, -tb, 0.0))
    parts.append(base_box.triangulate())

    # Scianka lewa (x od -tw do 0) i prawa (x od lx do lx + tw)
    left_wall = pv.Box(bounds=(-tw, 0.0, y_min_ext, y_max_ext, 0.0, hz_val))
    right_wall = pv.Box(bounds=(lx, lx + tw, y_min_ext, y_max_ext, 0.0, hz_val))
    parts.extend([left_wall.triangulate(), right_wall.triangulate()])

    # Scianka tylna naprzeciwko wlotu z kanalu
    if open_side == "ymax":
        back_wall = pv.Box(bounds=(0.0, lx, -tw, 0.0, 0.0, hz_val))
    else:
        back_wall = pv.Box(bounds=(0.0, lx, ly, ly + tw, 0.0, hz_val))
    parts.append(back_wall.triangulate())

    return pv.merge(parts)


def process_cavity(mesh_part, x0_lu, y0_lu, open_side, out_path):
    # Kopiowanie podsiatki pojedynczej wneki
    sub = mesh_part.extract_surface().triangulate().copy()
    pts = sub.points.copy()

    # Translacja do lokalnego ukladu wspolrzednych (0, 0) i przeskalowanie X, Y do milimetrow
    pts[:, 0] = (pts[:, 0] - x0_lu) * scale_xy
    pts[:, 1] = (pts[:, 1] - y0_lu) * scale_xy

    # Normalizacja osi Z do zakresu [-z_embed, hz] niezaleznie od pierwotnej grubosci STL
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    if z_max > z_min:
        pts[:, 2] = -z_embed + ((pts[:, 2] - z_min) / (z_max - z_min)) * (hz + z_embed)

    sub.points = pts

    # Dolaczenie plyty dennej i scianek kasety
    if ADD_CASSETTE:
        shell = build_cassette_shell(cav_lx, cav_ly, hz, t_base, t_wall, open_side)
        final_mesh = pv.merge([sub, shell]).extract_surface().triangulate()
    else:
        final_mesh = sub

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    final_mesh.save(out_path)
    print(f"Zapisano: {out_path} | Bounds [mm]: {np.round(final_mesh.bounds, 2)}")


# Lista modeli wejsciowych do przetworzenia
INPUT_MODELS = [
    ("pierwotna", "kp/shallow_water/thesis/print/basic_mesh_topology_00.stl"),
    ("oczyszczona", "kp/shallow_water/thesis/print/cleaned_topology_01.stl"),
]

out_dir = f"kp/shallow_water/thesis/print/modified/stl_druk_L{int(L_MM)}mm"

for tag, in_path in INPUT_MODELS:
    if not os.path.exists(in_path):
        print(f"Pominieto (brak pliku): {in_path}")
        continue

    raw_mesh = pv.read(in_path)

    # Podzial siatki wzgledem osi kanalu (Y = 154 LU) na wneke dolna i gorna
    bot_mesh = raw_mesh.clip(normal=(0, 1, 0), origin=(0, Y_SPLIT_LU, 0), invert=True)
    top_mesh = raw_mesh.clip(normal=(0, 1, 0), origin=(0, Y_SPLIT_LU, 0), invert=False)

    # Generowanie osobnych plikow STL dla obu wnek
    process_cavity(
        bot_mesh, X_MIN_LU, Y_BOT_MIN_LU, "ymax",
        f"{out_dir}/kaseta_{tag}_wneka_dolna_L{int(L_MM)}mm.stl"
    )
    process_cavity(
        top_mesh, X_MIN_LU, Y_TOP_MIN_LU, "ymin",
        f"{out_dir}/kaseta_{tag}_wneka_gorna_L{int(L_MM)}mm.stl"
    )