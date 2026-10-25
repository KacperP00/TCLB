import sys
import os
import math
import numpy as np
import pandas as pd

# Argumenty: 1: mnoznik skali, 2: bazowa dlugosc fali na siatce 1.0x [LU], 3: plik wyjsciowy CSV
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
base_lambda = float(sys.argv[2]) if len(sys.argv) > 2 else 324.0
out_path = sys.argv[3] if len(sys.argv) > 3 else f"kp/shallow_water/thesis/inlet_wave_{scale}x.csv"
base_iters = float(sys.argv[4]) if len(sys.argv) > 4 else 71000.0   # 4: dlugosc rekordu na siatce 1.0x (musi byc = SOLVE_ITERS)

g = 0.00209424
H = 9.55
A0 = 0.0625

iterations = int(math.floor(base_iters * scale + 0.5))
wave_length = base_lambda * scale

c_wave = np.sqrt(g * H)
period = wave_length / c_wave
omega = 2.0 * np.pi / period

t = np.arange(iterations)
# FIX-C1: gladki start (sin^2 przez 2 okresy) zamiast skoku do A0 w t=0 - usuwa szerokopasmowy transient po pierwszym froncie.
# Analiza (diagnostics.py) zaczyna sie 3 okresy po przyjsciu frontu, wiec rampa 2 okresy miesci sie w marginesie.
ramp_len = 2.0 * period
ramp = np.where(t < ramp_len, np.sin(0.5 * np.pi * t / ramp_len) ** 2, 1.0)
wave_h = A0 * ramp * np.cos(omega * t)

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
pd.DataFrame({"iter": t, "h_inlet": wave_h}).to_csv(out_path, index=False)
print(f"Wygenerowano fale ciagla ({scale}x, lambda={wave_length:.1f} LU, N={iterations}): {out_path}")