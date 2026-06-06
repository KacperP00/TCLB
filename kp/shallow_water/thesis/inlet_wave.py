import sys
import numpy as np
import pandas as pd

# --- PARAMETRY WEJŚCIOWE (Zgodne z fizyką kanału) ---
GRAVITY = 0.00209424
HEIGHT = 9.55
AMPLITUDE = 0.0625

# Odczyt długości fali (lambda) z argumentu przekazanego przez skrypt główny
if len(sys.argv) > 1:
    WAVE_LENGTH = float(sys.argv[1])
else:
    print("Błąd: Nie podano długości fali!")
    sys.exit(1)

# --- OBLICZENIA FIZYCZNE ---
c = np.sqrt(GRAVITY * HEIGHT)
period = WAVE_LENGTH / c
omega = 2.0 * np.pi / period

# --- GENEROWANIE FALI (SZTYWNY BUFOR NA 35000 ITERACJI) ---
t = np.arange(35000)
wave_h = AMPLITUDE * np.cos(omega * t)

df = pd.DataFrame({'iter': t, 'cos': wave_h})
df.to_csv('kp/shallow_water/thesis/inlet_wave.csv', index=False)

print(f"Wygenerowano inlet_wave.csv: lambda = {WAVE_LENGTH:.1f}, okres = {period:.1f}, omega = {omega:.6f}")