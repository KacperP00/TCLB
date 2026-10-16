import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# Parametry fizyczne LBM (niezmienne!)
g = 0.00209424
H = 9.55
c_wave = np.sqrt(g * H)
A0 = 0.0625

# Parametry czasu (Zeskalowane x1.5)
N = 106500
t = np.arange(N)
sigma = 4500
t_center = 5 * sigma

# Długość fali w jednostkach sieci rośnie 1.5-krotnie!
f_min = c_wave / 600.0  # Dawniej 400.0
f_max = c_wave / 300.0  # Dawniej 200.0

f_c = (f_min + f_max) / 2.0
omega_c = 2.0 * np.pi * f_c

gauss_window = np.exp(-0.5 * ((t - t_center) / sigma)**2)
carrier = np.cos(omega_c * (t - t_center))
h_inlet = A0 * gauss_window * carrier
h_inlet = h_inlet * (A0 / np.max(np.abs(h_inlet)))

# Zapis (nowa nazwa pliku!)
output_csv = 'kp/shallow_water/thesis/inlet_packet_1_5x.csv'
df = pd.DataFrame({'iter': t, 'h_inlet': h_inlet})
df.to_csv(output_csv, index=False)

print("Wygenerowano zaktualizowany pakiet falowy 1.5x.")