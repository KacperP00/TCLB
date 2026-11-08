import sys
import os
import numpy as np
import pandas as pd

# Parametry fizyczne układu
GRAVITY = 0.00209424
HEIGHT = 9.55
AMPLITUDE = 0.0625
ITERATIONS = 100000

# Wczytanie lambda z argumentu lub użycie wartości domyślnej (rezonans Euve et al.)
if len(sys.argv) > 1:
    WAVE_LENGTH = float(sys.argv[1])
else:
    WAVE_LENGTH = 294.0

# Kinematyka fali
c = np.sqrt(GRAVITY * HEIGHT)
period = WAVE_LENGTH / c
omega = 2.0 * np.pi / period

# Generowanie fali ciągłej
t = np.arange(ITERATIONS)
wave_h = AMPLITUDE * np.cos(omega * t)

# Eksport danych do pliku wejściowego
out_path = 'kp/shallow_water/thesis/inlet_wave.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df = pd.DataFrame({'iter': t, 'h_inlet': wave_h})
df.to_csv(out_path, index=False)

print(f"Wygenerowano {out_path}: lambda = {WAVE_LENGTH:.1f} LU, okres = {period:.1f} iter, omega = {omega:.6f}")
