import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import irfft, rfftfreq, fft

# Parametry modelu LBM
g = 0.00209424
H = 9.55
c_wave = np.sqrt(g * H)
A0 = 0.0625

# Parametry czasu
N = 31113
t = np.arange(N)
t_center = 15556

# Obliczanie granic pasma dla częstotliwości
f_min = c_wave / 400.0
f_max = c_wave / 200.0

# Generowanie idealnego widma w dziedzinie częstotliwości
freqs = rfftfreq(N, d=1.0)
spectrum_complex = np.zeros_like(freqs, dtype=complex)

band_mask = (freqs >= f_min) & (freqs <= f_max)
spectrum_amp = np.zeros_like(freqs)
spectrum_amp[band_mask] = 1.0  

phases = -2.0 * np.pi * freqs * t_center
spectrum_complex = spectrum_amp * np.exp(1j * phases)

# Transformata odwrotna (iFFT)
h_inlet = irfft(spectrum_complex, n=N)

# Aplikacja okna czasowego
window = np.exp(-0.5 * ((t - t_center) / 3000)**2)
h_inlet = h_inlet * window

# Normalizacja amplitudy
h_inlet = h_inlet * (A0 / np.max(np.abs(h_inlet)))

# Zapis do pliku wejściowego
output_csv = 'kp/shallow_water/thesis/inlet_wave.csv'
df = pd.DataFrame({'iter': t, 'h_inlet': h_inlet})
df.to_csv(output_csv, index=False)

# Obliczenia FFT dla sygnału z oknem
yf_real = np.abs(fft(h_inlet)) * (2.0 / N)
xf_real = rfftfreq(N, d=1.0)
mask_plot = (xf_real > 0)
f_plot = xf_real[mask_plot]
amps_real = yf_real[:len(f_plot)]

# Generowanie wykresów
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

# Sygnał w dziedzinie czasu
ax1.plot(t, h_inlet, 'b-')
ax1.set_title("Sygnał czasowy na wlocie do kanału")
ax1.set_xlabel("Iteracje (Czas)")
ax1.set_ylabel("Wychylenie lustra wody [LU]")
ax1.set_xlim(0, 31113)
ax1.grid(True, alpha=0.3)

# Sygnał w dziedzinie częstotliwości
ax2.plot(f_plot, amps_real, 'r-', linewidth=2, label='Widmo po oknie czasowym')
ax2.axvline(f_min, color='k', linestyle=':', label=r'$f_{min}$ (dla $\lambda=400$)')
ax2.axvline(f_max, color='k', linestyle=':', label=r'$f_{max}$ (dla $\lambda=200$)')
ax2.axvspan(f_min, f_max, color='gray', alpha=0.15, label='Zadane pasmo')

# Konfiguracja osi częstotliwości (Dół)
ax2.set_xlim(0.0001, 0.0012)
ax2.set_title("Spektrum w dziedzinie częstotliwości")
ax2.set_xlabel("Częstotliwość $f$ [1/iter]")
ax2.set_ylabel("Amplituda fali składowej")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Funkcje transformacji dla osi nieliniowej
def f2lam(f):
    return np.where(f == 0, np.nan, c_wave / f)

def lam2f(lam):
    return np.where(lam == 0, np.nan, c_wave / lam)

# Konfiguracja sprzężonej osi długości fali (Góra)
secax = ax2.secondary_xaxis('top', functions=(f2lam, lam2f))
secax.set_xlabel('Długość fali $\lambda$ [LU]')
secax.set_xticks([200, 250, 300, 400, 600, 1000])

plt.tight_layout()
plt.savefig('pakiet_ifft_dual_axis.png', dpi=300)
print("Wygenerowano raport: pakiet_ifft_dual_axis.png")