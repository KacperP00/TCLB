import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# Parametry środowiska LBM
g = 0.00209424
H = 9.55
L = 100.0
c_wave = np.sqrt(g * H)
A0 = 0.0625

# Parametry czasu
N = 31113
t = np.arange(N)
t_center = N // 2

# Definicja docelowego, szarego pasma badawczego
f_min = c_wave / 400.0
f_max = c_wave / 200.0

# Wyznaczenie fali nośnej (idealny środek pasma)
f_c = (f_min + f_max) / 2.0
omega_c = 2.0 * np.pi * f_c

# ---------------------------------------------------------
# GENEROWANIE SYGNAŁU: Klasyczny Cosinus w Oknie Gaussa
# ---------------------------------------------------------
# Zwiększenie sigmy zwęża spektrum częstotliwości. 
# Wartość 4200 gwarantuje, że widmo nie wyleje się poza zakres 10%.
sigma = 3000

# 1. Okno Gaussa (kształtuje gładki dzwon)
gauss_window = np.exp(-0.5 * ((t - t_center) / sigma)**2)
# 2. Nośna (wzbudza wodę z odpowiednią częstotliwością)
carrier = np.cos(omega_c * (t - t_center))

# Złożenie sygnału
h_inlet = A0 * gauss_window * carrier

# Normalizacja amplitudy (maksymalne wychylenie równe A0)
h_inlet = h_inlet * (A0 / np.max(np.abs(h_inlet)))

# Zapis warunku brzegowego
output_csv = 'kp/shallow_water/thesis/inlet_packet.csv'
df = pd.DataFrame({'iter': t, 'h_inlet': h_inlet})
df.to_csv(output_csv, index=False)

# ---------------------------------------------------------
# WERYFIKACJA WIDMA (ZERO-PADDING DLA GŁADKIEGO WYKRESU)
# ---------------------------------------------------------
N_plot = N * 8
yf = np.abs(rfft(h_inlet, n=N_plot)) * (2.0 / N)
xf = rfftfreq(N_plot, d=1.0)

mask = (xf > 0)
f_plot = xf[mask]
amps_real = yf[mask]

# Generowanie wykresów
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

# Wykres domeny czasu
ax1.plot(t, h_inlet, 'b-')
ax1.set_title(f"Sygnał wejściowy LBM (Sigma = {sigma})")
ax1.set_xlabel("Iteracja")
ax1.set_ylabel("Wychylenie [LU]")
ax1.set_xlim(0, N)
ax1.grid(True, alpha=0.3)

# Wykres domeny częstotliwości
ax2.plot(f_plot, amps_real, 'r-', linewidth=2, label='Widmo sygnału')
ax2.axvline(f_min, color='k', linestyle=':', label='Dolna granica ($\lambda=400$)')
ax2.axvline(f_max, color='k', linestyle=':', label='Górna granica ($\lambda=200$)')
ax2.axvline(f_c, color='g', linestyle='--', label='Fala nośna (Środek pasma)')
ax2.axvspan(f_min, f_max, color='gray', alpha=0.15, label='Zadane pasmo')

ax2.set_xlim(0.0001, 0.0012)
ax2.set_title("Spektrum względem częstotliwości")
ax2.set_xlabel("Częstotliwość $f$ [1/iter]")
ax2.set_ylabel("Amplituda widma")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bezpieczna transformacja nieliniowej osi sprzężonej
def f2lam(f):
    f_safe = np.where(np.asarray(f) < 1e-8, 1e-8, np.asarray(f))
    return c_wave / f_safe

def lam2f(lam):
    lam_safe = np.where(np.asarray(lam) < 1e-8, 1e-8, np.asarray(lam))
    return c_wave / lam_safe

secax = ax2.secondary_xaxis('top', functions=(f2lam, lam2f))
secax.set_xlabel('Długość fali $\lambda$ [LU]')
secax.set_xticks([200, 250, 300, 400, 600, 1000])

plt.tight_layout()
plt.savefig('kp/shallow_water/thesis/sygnal_spektrum_gauss.png', dpi=300)
print("Raport wygenerowany. Spektrum zwężone.")