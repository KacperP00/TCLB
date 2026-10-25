import sys
import os

# Argumenty: 1: mnoznik skali, 2: szablon .tpl, 3: plik wyjsciowy .xml, 4: katalog output VTI, 5: plik CSV, 6: plik STL
scale = float(sys.argv[1])
tpl_path = sys.argv[2]
out_xml_path = sys.argv[3]
out_dir = sys.argv[4]
csv_file = sys.argv[5]
stl_file = sys.argv[6] if len(sys.argv) > 6 else ""

def sc(val):
    # Skalowanie wymiaru siatki z zaokragleniem do liczby calkowitej
    return int(round(val * scale))

nx = sc(8416)
ds_dx = sc(4000)
ds_nx = sc(400)
wall_after_dx = ds_dx + ds_nx

# Wyliczenie wspolrzednych wnek Euve bez ryzyka szczelin zaokraglen
euve_w2_nx = sc(4154)
euve_c_nx = sc(44)
euve_sep_nx = sc(4)
euve_c1_dx = euve_w2_nx
euve_sep_dx = euve_c1_dx + euve_c_nx
euve_c2_dx = euve_sep_dx + euve_sep_nx
euve_w6_dx = euve_c2_dx + euve_c_nx
euve_w6_nx = nx - euve_w6_dx

params = {
    "OUT_DIR": out_dir,
    "CSV_FILE": csv_file,
    "STL_FILE": stl_file,
    "NX": nx,
    "NY": sc(308),
    "INLET_DY": sc(104),
    "INLET_NY": sc(100),
    "OUTLET_DX": sc(8400),
    "OUTLET_NX": nx - sc(8400),
    "WALL_NY": sc(104),
    "WALL_TOP_DY": sc(204),
    "DS_DX": ds_dx,
    "DS_NX": ds_nx,
    "DS1_DY": sc(4),
    "DS2_DY": sc(204),
    "DS_NY": sc(100),
    "WALL_AFTER_DX": wall_after_dx,
    "WALL_AFTER_NX": sc(8408) - wall_after_dx,
    "WALL_BACK2_DY": sc(304),
    "OBJ1_DX": sc(4600),
    "OBJ2_DX": sc(1500),
    "OBJ_NX": sc(400),
    "OBJ_DY": sc(154),
    "OBJ_NY": sc(8),
    "EUVE_W2_NX": euve_w2_nx,
    "EUVE_C1_DX": euve_c1_dx,
    "EUVE_C_NX": euve_c_nx,
    "EUVE_C1_NY": sc(104) - sc(54),
    "EUVE_SEP_DX": euve_sep_dx,
    "EUVE_SEP_NX": euve_sep_nx,
    "EUVE_C2_DX": euve_c2_dx,
    "EUVE_C2_NY": sc(104) - sc(61),
    "EUVE_W6_DX": euve_w6_dx,
    "EUVE_W6_NX": euve_w6_nx,
    "NU": f"{0.000228 * scale:.8f}",
    "ADJ_START1": sc(34000),
    "ADJ_START2": sc(42000),
    "ADJ_END": sc(65000),
    "SOLVE_ITERS": sc(71000),
    "VTK_ITERS": sc(200),
    "LOG_ITERS": sc(1000),
}

with open(tpl_path, "r", encoding="utf-8") as f:
    content = f.read()

with open(out_xml_path, "w", encoding="utf-8") as f:
    f.write(content.format(**params))