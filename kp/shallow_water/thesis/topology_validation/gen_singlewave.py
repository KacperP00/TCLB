import sys
import os
import numpy as np
import pandas as pd

# Argumenty: 1: mnoznik skali, 2: bazowa dlugosc fali na siatce 1.0x [LU], 3: plik wyjsciowy CSV
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
base_lambda = float(sys.argv[2]) if len(sys.argv) > 2 else 324.0
out_path = sys.argv[3] if len(sys.argv) > 3 else f"kp/shallow_water/thesis/inlet_wave_{scale}x.csv"

g = 0.00209424
H = 9.55
A0 = 0.0625

iterations = int(round(71000 * scale))
wave_length = base_lambda * scale

c_wave = np.sqrt(g * H)
period = wave_length / c_wave
omega = 2.0 * np.pi / period

t = np.arange(iterations)
wave_h = A0 * np.cos(omega * t)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
pd.DataFrame({"iter": t, "h_inlet": wave_h}).to_csv(out_path, index=False)
print(f"Wygenerowano fale ciagla ({scale}x, lambda={wave_length:.1f} LU, N={iterations}): {out_path}")