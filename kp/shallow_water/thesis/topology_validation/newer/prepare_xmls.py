import sys
import os
import math

# Argumenty: 1: mnoznik skali, 2: szablon .tpl, 3: plik wyjsciowy .xml, 4: katalog output VTI, 5: plik CSV, 6: plik STL
scale = float(sys.argv[1])
tpl_path = sys.argv[2]
out_xml_path = sys.argv[3]
out_dir = sys.argv[4]
csv_file = sys.argv[5]
stl_file = sys.argv[6] if len(sys.argv) > 6 else ""
base_iters = float(sys.argv[7]) if len(sys.argv) > 7 else 65000.0   # 7: dlugosc rekordu na siatce 1.0x

def sc(val):
    # FIX-A1: zaokraglanie 'half-up' (round() w Pythonie zaokragla .5 do parzystej, np. round(62.5)=62 ale round(91.5)=92)
    return int(math.floor(val * scale + 0.5))

# FIX-A2: niezmienniki geometrii - jesli ktorys nie zachodzi, skala jest niedopuszczalna
# (np. scale=1.1 daje NY=339, a sciany koncza sie na y=338 -> otwarty rzad plynu na brzegu domeny)
_inv = {
    "kanal: dy_dolna_sciana + szerokosc = dy_gorna_sciana": sc(104) + sc(100) == sc(204),
    "gorna sciana domyka domene: 204 + 104 = 308":          sc(204) + sc(104) == sc(308),
    "wneka dolna miesci sie w kanale: 4 + 100 = 104":        sc(4) + sc(100) == sc(104),
    "wneka gorna: 204 + 100 = 304":                          sc(204) + sc(100) == sc(304),
    "tylna sciana gornej wneki: 304 + 4 = 308":              sc(304) + sc(4) == sc(308),
    "wneka nie wychodzi poza domene w X":                    sc(4000) + sc(400) < sc(8416),
}
_bad = [k for k, ok in _inv.items() if not ok]
if _bad:
    sys.exit("BLAD: skala %g psuje geometrie (zaokraglenia): %s\nUzywaj skal, dla ktorych 104*s, 100*s, 4*s sa calkowite (np. 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)." % (scale, "; ".join(_bad)))

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
    "SOLVE_ITERS": sc(base_iters),
    "VTK_ITERS": sc(200),
    "LOG_ITERS": sc(1000),
}

with open(tpl_path, "r", encoding="utf-8") as f:
    content = f.read()

with open(out_xml_path, "w", encoding="utf-8") as f:
    f.write(content.format(**params))