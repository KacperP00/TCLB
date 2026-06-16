import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq

# Inicjalizacja parametrów fizycznych 
xmax = 10
L = 0.16
g = 9.81
H = 0.02

# Parametry fali nośnej i dyskretyzacja w czasie
k = (0 + 1) * np.pi / (2 * L)
omega = k * np.sqrt(g * H)
T0 = 2 * np.pi / omega
dt = 0.0288 * np.sqrt(H / g)
time = np.arange(0, 4 * T0 + dt/2, dt)

# Transformacja do wartości bezwymiarowych
omega_nd = omega / np.sqrt(g / H)
h_nd = H / L
k_nd = k * L
time_nd = time / np.sqrt(H / g)
sigma_t = 0.4 * 2 / np.sqrt(H / g)

# Wyznaczenie przesunięcia czasowego do środka przedziału
t0_idx = len(time_nd) // 2
t0 = time_nd[t0_idx]

# Generowanie sygnału w domenie czasu
f_signal = 1 * np.real(
    np.exp(-0.5 * ((time_nd - t0) / sigma_t)**2) * np.exp(1j * omega_nd * time_nd) * np.exp(1j * k_nd * (-xmax))
)

# Rysowanie wykresu w dziedzinie czasu
plt.figure()
plt.plot(f_signal)
plt.xlabel('t')
plt.ylabel('signal')
plt.savefig('matlab_signal_time.png', dpi=300)
plt.show()

# Obliczenia parametrów dla transformaty
dt_nondim = time_nd[1] - time_nd[0]
frame_rate = 1 / dt_nondim
freq = omega_nd / (2 * np.pi)
Np = len(f_signal)

# Transformata Fouriera z rozszerzeniem zerami do 50000 próbek
OnePointFFT = 2 * fftshift(fft(f_signal, n=50000)) / Np

# Wyznaczenie wektora częstotliwości i bezwymiarowej liczby falowej
freqs = fftshift(fftfreq(50000, d=dt_nondim))
omegas_nondim = freqs * (2 * np.pi)
ks = omegas_nondim / h_nd

# Rysowanie wykresu widma
plt.figure()
plt.plot(ks / np.pi, np.abs(OnePointFFT))
plt.xlabel('$k_{nondim}$')
plt.ylabel('|A|')
plt.xlim([0, 1])
plt.savefig('matlab_signal_spectrum.png', dpi=300)
plt.show()