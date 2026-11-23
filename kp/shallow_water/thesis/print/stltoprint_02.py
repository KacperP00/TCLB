import os
import sys
import numpy as np
import pyvista as pv

# Parametry stanowiska: szerokosc kanalu L [mm], glebokosc wody [mm], zapas nad woda [mm], uskok dna [mm]
L_MM = float(sys.argv[1]) if len(sys.argv) > 1 else 160.0
H_WATER_MM = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
H_CLEARANCE_MM = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
H_STEP_MM = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0

# Parametry konstrukcyjne druku: grubosc scianek/plyty [mm], szerokosc kolnierza [mm], liczba modulow w X
T_PLATE_MM = 4.0
T_WALL_MM = 4.0
FLANGE_MM = 20.0
N_SEGMENTS = 4
Z_EMBED = 0.8

# Stale geometryczne siatki bazowej 1.0x w jednostkach LU
L_LU = 100.0
X_MIN_LU = 4000.0
Y_SPLIT_LU = 154.0
Y_BOT_MIN_LU = 4.0
Y_TOP_MIN_LU = 204.0

# Przeliczenie wymiarow w milimetrach
scale_xy = L_MM / L_LU
cav_lx = 4.0 * L_MM
cav_ly = 1.0 * L_MM
h_active = H_WATER_MM + H_CLEARANCE_MM


def make_box(xmin, xmax, ymin, ymax, zmin, zmax):
    # Utworzenie zamkniętej siatki prostopadloscianu o zadanych granicach
    return pv.Box(bounds=(xmin, xmax, ymin, ymax, zmin, zmax)).extract_surface().triangulate()


def normalize_islands(mesh_part, x0_lu, y0_lu, z_bottom, z_top):
    # Przesuniecie wysp do lokalnego ukladu (0,0), przeskalowanie XY i rozciagniecie w osi Z do [z_bottom, z_top]
    sub = mesh_part.extract_surface().triangulate().copy()
    pts = sub.points.copy()
    pts[:, 0] = (pts[:, 0] - x0_lu) * scale_xy
    pts[:, 1] = (pts[:, 1] - y0_lu) * scale_xy
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    if z_max > z_min:
        pts[:, 2] = z_bottom + ((pts[:, 2] - z_min) / (z_max - z_min)) * (z_top - z_bottom)
    sub.points = pts
    return sub.clean()


def build_top_hang_cavity(mesh_part, x0_lu, y0_lu, open_side, islands_height):
    # Wariant 1 (Promotor): plyta laczaca powyzej powierzchni wody + scianki + kolnierz boczny, obrot 180 deg do druku
    isl = normalize_islands(mesh_part, x0_lu, y0_lu, 0.0, islands_height + Z_EMBED)
    tw, tp, fl = T_WALL_MM, T_PLATE_MM, FLANGE_MM

    # Zakres Y obejmujacy scianke tylna i kolnierz opierajacy sie na scianie kanalu
    y_min_roof = -(tw + fl) if open_side == "ymax" else 0.0
    y_max_roof = cav_ly if open_side == "ymax" else cav_ly + tw + fl
    y_min_wall = -tw if open_side == "ymax" else 0.0
    y_max_wall = cav_ly if open_side == "ymax" else cav_ly + tw

    # Plyta gorna (strop ponad woda: z od islands_height do islands_height + tp)
    roof = make_box(-tw, cav_lx + tw, y_min_roof, y_max_roof, islands_height, islands_height + tp)

    # Scianki boczne (lewa i prawa) oraz tylna wpuszczone w plyte gorna na Z_EMBED
    left_w = make_box(-tw, 0.0, y_min_wall, y_max_wall, 0.0, islands_height + Z_EMBED)
    right_w = make_box(cav_lx, cav_lx + tw, y_min_wall, y_max_wall, 0.0, islands_height + Z_EMBED)
    if open_side == "ymax":
        back_w = make_box(-Z_EMBED, cav_lx + Z_EMBED, -tw, 0.0, 0.0, islands_height + Z_EMBED)
    else:
        back_w = make_box(-Z_EMBED, cav_lx + Z_EMBED, cav_ly, cav_ly + tw, 0.0, islands_height + Z_EMBED)

    combined = pv.merge([isl, roof, left_w, right_w, back_w]).extract_surface().triangulate()

    # Obrot o 180 stopni wokol osi X: plyta gorna laduje na stole drukarki (Z=0), wyspy rosna w gore (0% podpor)
    z_center = 0.5 * (islands_height + tp)
    y_center = 0.5 * cav_ly
    rotated = combined.rotate_x(180.0, point=(0.0, y_center, z_center), inplace=False)
    return rotated.clean()


def build_top_bridge_both(bot_part, top_part, islands_height):
    # Wariant 1B (Promotor): obie wneki polaczone ponad woda sztywnymi mostkami nad kanalem glownym
    isl_bot = normalize_islands(bot_part, X_MIN_LU, Y_BOT_MIN_LU, 0.0, islands_height + Z_EMBED)
    isl_top = normalize_islands(top_part, X_MIN_LU, Y_TOP_MIN_LU, 0.0, islands_height + Z_EMBED)
    # Przesuniecie gornej wneki o szerokosc dolnej wneki (cav_ly) + szerokosc kanalu (L_MM) = 2 * L_MM
    isl_top.points[:, 1] += 2.0 * L_MM

    tw, tp = T_WALL_MM, T_PLATE_MM
    y_top_start = 2.0 * L_MM
    y_top_end = 3.0 * L_MM

    # Stropy nad dolna i gorna wneka
    roof_bot = make_box(-tw, cav_lx + tw, -tw, cav_ly, islands_height, islands_height + tp)
    roof_top = make_box(-tw, cav_lx + tw, y_top_start, y_top_end + tw, islands_height, islands_height + tp)

    # Scianki zewnetrzne dolnej i gornej wneki
    w_bot_l = make_box(-tw, 0.0, -tw, cav_ly, 0.0, islands_height + Z_EMBED)
    w_bot_r = make_box(cav_lx, cav_lx + tw, -tw, cav_ly, 0.0, islands_height + Z_EMBED)
    w_bot_b = make_box(-Z_EMBED, cav_lx + Z_EMBED, -tw, 0.0, 0.0, islands_height + Z_EMBED)

    w_top_l = make_box(-tw, 0.0, y_top_start, y_top_end + tw, 0.0, islands_height + Z_EMBED)
    w_top_r = make_box(cav_lx, cav_lx + tw, y_top_start, y_top_end + tw, 0.0, islands_height + Z_EMBED)
    w_top_b = make_box(-Z_EMBED, cav_lx + Z_EMBED, y_top_end, y_top_end + tw, 0.0, islands_height + Z_EMBED)

    # Belki mostkowe laczace obie wneki ponad powierzchnia wody w kanale (po 2 na kazdy modul)
    bridges = []
    beam_w = 24.0
    step_x = cav_lx / N_SEGMENTS
    for i in range(N_SEGMENTS):
        x_s = i * step_x
        x_e = (i + 1) * step_x
        b1 = make_box(x_s, x_s + beam_w, cav_ly - Z_EMBED, y_top_start + Z_EMBED, islands_height, islands_height + tp)
        b2 = make_box(x_e - beam_w, x_e, cav_ly - Z_EMBED, y_top_start + Z_EMBED, islands_height, islands_height + tp)
        bridges.extend([b1, b2])

    parts = [isl_bot, isl_top, roof_bot, roof_top, w_bot_l, w_bot_r, w_bot_b, w_top_l, w_top_r, w_top_b] + bridges
    combined = pv.merge(parts).extract_surface().triangulate()

    # Obrot o 180 stopni wokol osi X (mostki i stropy leza plasko na stole drukarki Z=0)
    z_center = 0.5 * (islands_height + tp)
    y_center = 1.5 * L_MM
    return combined.rotate_x(180.0, point=(0.0, y_center, z_center), inplace=False).clean()


def build_raised_bottom_cavity(mesh_part, x0_lu, y0_lu, open_side):
    # Wariant 2 (dr Daria): cokol denny o wysokosci H_STEP_MM (+30 mm) + wyspy o wysokosci h_active (30 mm)
    isl = normalize_islands(mesh_part, x0_lu, y0_lu, H_STEP_MM - Z_EMBED, H_STEP_MM + h_active)
    tw = T_WALL_MM
    h_tot = H_STEP_MM + h_active

    y_min_ext = -tw if open_side == "ymax" else 0.0
    y_max_ext = cav_ly if open_side == "ymax" else cav_ly + tw

    # Cokol denny (z od 0 do H_STEP_MM) wyrownujacy uskok +3 cm w kanale
    base_pedestal = make_box(-tw, cav_lx + tw, y_min_ext, y_max_ext, 0.0, H_STEP_MM)

    # Scianki obwodowe prowadzone od samego spodu (z = 0) do pelnej wysokosci (h_tot)
    left_w = make_box(-tw, 0.0, y_min_ext, y_max_ext, 0.0, h_tot)
    right_w = make_box(cav_lx, cav_lx + tw, y_min_ext, y_max_ext, 0.0, h_tot)
    if open_side == "ymax":
        back_w = make_box(-Z_EMBED, cav_lx + Z_EMBED, -tw, 0.0, 0.0, h_tot)
    else:
        back_w = make_box(-Z_EMBED, cav_lx + Z_EMBED, cav_ly, cav_ly + tw, 0.0, h_tot)

    return pv.merge([isl, base_pedestal, left_w, right_w, back_w]).extract_surface().triangulate().clean()


def clip_closed_x(mesh, x_min_cut, x_max_cut):
    # Wyciecie modulu wzdluz osi X z automatycznym zamknieciem przekrojow (bryla zamknieta manifold)
    res = mesh.copy()
    try:
        res = res.clip_closed_surface(normal=(1, 0, 0), origin=(x_min_cut, 0, 0))
        res = res.clip_closed_surface(normal=(-1, 0, 0), origin=(x_max_cut, 0, 0))
    except Exception:
        res = res.clip(normal=(1, 0, 0), origin=(x_min_cut, 0, 0), invert=False)
        res = res.clip(normal=(-1, 0, 0), origin=(x_max_cut, 0, 0), invert=False)
    return res.extract_surface().triangulate().clean()


def save_full_and_segments(mesh, base_out_path):
    # Zapis pelnego modelu (640 mm) oraz 4 gotowych do druku modulow (po 160 mm) z wyzerowanym X
    os.makedirs(os.path.dirname(base_out_path) or ".", exist_ok=True)
    mesh.save(base_out_path)
    print(f"Zapisano pelny model: {base_out_path} | Bounds [mm]: {np.round(mesh.bounds, 1)}")

    seg_dir = os.path.join(os.path.dirname(base_out_path), "moduly_do_druku_160mm")
    os.makedirs(seg_dir, exist_ok=True)
    step_x = cav_lx / N_SEGMENTS
    stem = os.path.splitext(os.path.basename(base_out_path))[0]

    for i in range(N_SEGMENTS):
        x0 = -T_WALL_MM - 0.1 if i == 0 else i * step_x
        x1 = cav_lx + T_WALL_MM + 0.1 if i == N_SEGMENTS - 1 else (i + 1) * step_x
        seg = clip_closed_x(mesh, x0, x1)
        if seg.n_points > 0:
            # Przesuniecie wycietego modulu do X=0 dla wygodnego ulozenia na stole drukarki
            seg.points[:, 0] -= seg.points[:, 0].min()
            seg_path = os.path.join(seg_dir, f"{stem}_modul_{i+1}z{N_SEGMENTS}.stl")
            seg.save(seg_path)


INPUT_MODELS = [
    ("pierwotna", "kp/shallow_water/thesis/print/basic_mesh_topology_00.stl"),
    ("oczyszczona", "kp/shallow_water/thesis/print/cleaned_topology_01.stl"),
]

out_root = f"kp/shallow_water/thesis/print/modified/stl_druk_L{int(L_MM)}mm"

for tag, in_path in INPUT_MODELS:
    if not os.path.exists(in_path):
        print(f"Pominieto (brak pliku): {in_path}")
        continue

    raw_mesh = pv.read(in_path)
    bot_mesh = raw_mesh.clip(normal=(0, 1, 0), origin=(0, Y_SPLIT_LU, 0), invert=True)
    top_mesh = raw_mesh.clip(normal=(0, 1, 0), origin=(0, Y_SPLIT_LU, 0), invert=False)

    # 1A. Wariant Promotora: nakladka wieszana od gory (wyspy 30 mm) - osobno lewa/dolna i prawa/gorna wneka
    m_top_bot30 = build_top_hang_cavity(bot_mesh, X_MIN_LU, Y_BOT_MIN_LU, "ymax", h_active)
    m_top_top30 = build_top_hang_cavity(top_mesh, X_MIN_LU, Y_TOP_MIN_LU, "ymin", h_active)
    save_full_and_segments(m_top_bot30, f"{out_root}/1_nakladka_gorna_30mm/nakladka_{tag}_wneka_dolna.stl")
    save_full_and_segments(m_top_top30, f"{out_root}/1_nakladka_gorna_30mm/nakladka_{tag}_wneka_gorna.stl")

    # 1B. Wariant Promotora: obie wneki spiete wspolnymi mostkami nad kanalem (do podgladu w NX i ciecia)
    m_bridge = build_top_bridge_both(bot_mesh, top_mesh, h_active)
    save_full_and_segments(m_bridge, f"{out_root}/1_nakladka_gorna_30mm/mostek_obie_wneki_{tag}.stl")

    # 2. Wariant dr Darii: kaseta z pelnym cokołem dennym +30 mm i wyspami 30 mm (wysokosc calkowita 60 mm)
    m_ped_bot = build_raised_bottom_cavity(bot_mesh, X_MIN_LU, Y_BOT_MIN_LU, "ymax")
    m_ped_top = build_raised_bottom_cavity(top_mesh, X_MIN_LU, Y_TOP_MIN_LU, "ymin")
    save_full_and_segments(m_ped_bot, f"{out_root}/2_dno_plus_30mm/kaseta_dno30_{tag}_wneka_dolna.stl")
    save_full_and_segments(m_ped_top, f"{out_root}/2_dno_plus_30mm/kaseta_dno30_{tag}_wneka_gorna.stl")