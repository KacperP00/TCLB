import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import irfft, rfftfreq, rfft
from scipy.signal.windows import tukey

# Parametry LBM
g = 0.00209424
H = 9.55
L = 100.0
c_wave = np.sqrt(g * H)
A0 = 0.0625

# Parametry czasu
N = 31113
t = np.arange(N)
t_center = N // 2

# Konstrukcja widma w dziedzinie kL/pi
freqs = rfftfreq(N, d=1.0)
k_axis = (2.0 * freqs * L) / c_wave
spectrum_amp = np.zeros_like(k_axis)

# Generowanie gładkiego spektrum na pasmie
for i, k in enumerate(k_axis):
    if k <= 0.5 or k >= 1.0:
        spectrum_amp[i] = 0.0
    elif 0.6 <= k <= 0.9:
        spectrum_amp[i] = 1.0
    elif 0.5 < k < 0.6:
        spectrum_amp[i] = 0.5 * (1 - np.cos(np.pi * (k - 0.5) / 0.1))
    elif 0.9 < k < 1.0:
        spectrum_amp[i] = 0.5 * (1 + np.cos(np.pi * (k - 0.9) / 0.1))

# Przesunięcie w czasie
phases = -2.0 * np.pi * freqs * t_center
spectrum_complex = spectrum_amp * np.exp(1j * phases)

# iFFT
h_inlet = irfft(spectrum_complex, n=N)

# Wygaszenie 5% sygnału na końcach
#window_time = tukey(N, alpha=0.05)
#h_inlet = h_inlet * window_time

# Normalizacja
h_inlet = h_inlet * (A0 / np.max(np.abs(h_inlet)))

# Zapis do CSV
output_csv = 'kp/shallow_water/thesis/inlet_wave.csv'
df = pd.DataFrame({'iter': t, 'h_inlet': h_inlet})
df.to_csv(output_csv, index=False)

# Weryfikacja FFT (zagęszczenie siatki x8 dla gładkiego wykresu)
N_plot = N * 8
yf_plot = np.abs(rfft(h_inlet, n=N_plot)) * (2.0 / N)
xf_plot = rfftfreq(N_plot, d=1.0)

# Oś wykresu
kL_pi_plot = (2.0 * xf_plot * L) / c_wave
mask_plot = (kL_pi_plot <= 1.5)
kL_pi_plot = kL_pi_plot[mask_plot]
amps_real = yf_plot[mask_plot]

# Wykresy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

# Sygnał czasowy
ax1.plot(t, h_inlet, 'b-')
ax1.set_title("Sygnał czasowy na wlocie (31 113 iteracji)")
ax1.set_xlabel("Iteracje (Czas)")
ax1.set_ylabel("Wychylenie [LU]")
ax1.set_xlim(0, N)
ax1.grid(True, alpha=0.3)

# Spektrum
ax2.plot(kL_pi_plot, amps_real, 'r-', linewidth=2, label='Rzeczywiste spektrum paczki')
ax2.axvline(0.5, color='k', linestyle='--', label='Dolna granica $kL/\pi = 0.5$')
ax2.axvline(1.0, color='k', linestyle='--', label='Górna granica $kL/\pi = 1.0$')
ax2.axvspan(0.5, 1.0, color='gray', alpha=0.15, label='Zadane pasmo badawcze')
ax2.set_xlim(0.0, 1.5)
ax2.set_title("Spektrum $kL/\pi$ (Gładkie zbocza i precyzyjne osadzenie)")
ax2.set_xlabel("Bezwymiarowa liczba falowa $kL/\pi$")
ax2.set_ylabel("Amplituda widma")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bezpieczna konwersja osi (ochrona przed wartościami zerowymi)
def kl2lam(kl):
    kl_safe = np.where(np.asarray(kl) < 1e-5, 1e-5, np.asarray(kl))
    return (2 * L) / kl_safe

def lam2kl(lam):
    lam_safe = np.where(np.asarray(lam) < 1e-5, 1e-5, np.asarray(lam))
    return (2 * L) / lam_safe

# Sprzężona oś długości fali
secax = ax2.secondary_xaxis('top', functions=(kl2lam, lam2kl))
secax.set_xlabel('Długość fali $\lambda$ [LU]')
secax.set_xticks([150, 200, 250, 300, 400, 800])

plt.tight_layout()
plt.savefig('pakiet_idealny.png', dpi=300)