import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parametry środowiska LBM
g = 0.00209424
H = 9.55
L = 100.0
A0 = 0.0625

# Wektor czasu dla symulacji
N = 31113
t = np.arange(N)
t0 = N // 2

# Wyznaczenie parametrów fali nośnej
c = np.sqrt(g * H)
k = 0.75 * np.pi / L  # Nośna w kL/pi = 0.75 (środek okna 0.5 - 1.0)
omega = k * c

# Generowanie sygnału wejściowego
sigma = 2400
h_inlet = A0 * np.cos(omega * (t - t0)) * np.exp(-0.5 * ((t - t0) / sigma)**2)

# Zapis warunku brzegowego
df = pd.DataFrame({'iter': t, 'h_inlet': h_inlet})
df.to_csv('inlet_wave.csv', index=False)

# Analiza spektralna (FFT)
yf = np.abs(fft(h_inlet)) * (2.0 / N)
xf = fftfreq(N, d=1.0)

mask = (xf > 0)
f_plot = xf[mask]
amps_real = yf[mask]

# Transformacja osi do domeny kL/pi
kL_pi_plot = (2.0 * f_plot * L) / c
mask_kl = (kL_pi_plot <= 1.5)
kL_pi_plot = kL_pi_plot[mask_kl]
amps_real = amps_real[mask_kl]

# Konstrukcja wykresów
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

# Domena czasu
ax1.plot(t, h_inlet, 'b-')
ax1.set_title(f"Sygnał wejściowy LBM (N = {N} iteracji)")
ax1.set_xlabel("Iteracja")
ax1.set_ylabel("Wychylenie [LU]")
ax1.set_xlim(0, N)
ax1.grid(True, alpha=0.3)

# Domena bezwymiarowej liczby falowej
ax2.plot(kL_pi_plot, amps_real, 'r-', linewidth=2, label='Widmo paczki Gaussa')
ax2.axvline(0.5, color='k', linestyle=':', label='Dolna granica $kL/\pi = 0.5$')
ax2.axvline(1.0, color='k', linestyle=':', label='Górna granica $kL/\pi = 1.0$')
ax2.axvline(0.75, color='g', linestyle='--', label='Fala nośna $kL/\pi = 0.75$')
ax2.axvspan(0.5, 1.0, color='gray', alpha=0.15, label='Pasmo')

ax2.set_xlim(0.0, 1.5)
ax2.set_title("Spektrum względem bezwymiarowej liczby falowej")
ax2.set_xlabel("kL/\pi")
ax2.set_ylabel("Amplituda")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Definicje funkcji dla sprzężonej osi nieliniowej
def kl2lam(kl):
    kl_safe = np.where(np.asarray(kl) < 1e-8, 1e-8, np.asarray(kl))
    return (2 * L) / kl_safe

def lam2kl(lam):
    lam_safe = np.where(np.asarray(lam) < 1e-8, 1e-8, np.asarray(lam))
    return (2 * L) / lam_safe

# Rysowanie osi długości fali
secax = ax2.secondary_xaxis('top', functions=(kl2lam, lam2kl))
secax.set_xlabel('Długość fali $\lambda$ [LU]')
secax.set_xticks([150, 200, 250, 300, 400, 800])

plt.tight_layout()
plt.savefig('pakiet_lbm_nosna_075.png', dpi=300)