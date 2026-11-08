import sys
import os
import math
import numpy as np
import pandas as pd

# Argumenty: 1: mnoznik skali, 2: sciezka wyjsciowa CSV
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
output_csv = sys.argv[2] if len(sys.argv) > 2 else f"kp/shallow_water/thesis/inlet_packet_{scale}x.csv"
base_iters = float(sys.argv[3]) if len(sys.argv) > 3 else 71000.0   # 3: dlugosc rekordu na siatce 1.0x (musi byc = SOLVE_ITERS)

g = 0.00209424
H = 9.55
c_wave = np.sqrt(g * H)
A0 = 0.0625

# Parametry czasu i pasma skalowane liniowo z rozdzielczoscia siatki
N = int(math.floor(base_iters * scale + 0.5))
t = np.arange(N)
sigma = 3000.0 * scale
t_center = 5.0 * sigma

f_min = c_wave / (400.0 * scale)
f_max = c_wave / (200.0 * scale)
f_c = 0.5 * (f_min + f_max)
omega_c = 2.0 * np.pi * f_c

gauss_window = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
carrier = np.cos(omega_c * (t - t_center))
h_inlet = A0 * gauss_window * carrier
h_inlet = h_inlet * (A0 / np.max(np.abs(h_inlet)))

os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
pd.DataFrame({"iter": t, "h_inlet": h_inlet}).to_csv(output_csv, index=False)
print(f"Wygenerowano paczke falowa ({scale}x, N={N}, sigma={sigma:.0f}): {output_csv}")