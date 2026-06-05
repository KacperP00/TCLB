#!/usr/bin/env python3
"""
Analyze wave parameters from simulation data
Extracts: k_real, k_imag, phase, frequency, amplitude, wavelength
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

def analyze_inlet_wave():
    """Analyze the inlet wave driving function"""
    print("="*70)
    print("INLET WAVE ANALYSIS (Boundary Condition)")
    print("="*70)
    
    inlet_csv = '/home/franek/TCLB/kp/shallow_water/thesis/inlet_wave.csv'
    df_inlet = pd.read_csv(inlet_csv)
    
    times = df_inlet['iter'].values
    amplitudes = df_inlet['cos'].values
    
    # Amplitude
    A = np.max(np.abs(amplitudes))
    
    # Frequency from FFT
    freqs = fftfreq(len(times), d=(times[1]-times[0]))
    fft_vals = np.abs(fft(amplitudes))
    pos_freqs = freqs[:len(freqs)//2]
    pos_fft = fft_vals[:len(fft_vals)//2]
    peak_idx = np.argmax(pos_fft[1:]) + 1
    peak_freq = pos_freqs[peak_idx]
    w = 2 * np.pi * peak_freq
    
    # Period
    period = 1 / peak_freq if peak_freq > 0 else np.inf
    
    print(f"Amplitude (A): {A:.8f}")
    print(f"Angular frequency (w): {w:.8f} [1/LU]")
    print(f"Period (T): {period:.2f} [LU]")
    print(f"Frequency (f): {peak_freq:.8f} [1/LU]")
    
    return {'A': A, 'w': w, 'period': period, 'freq': peak_freq}

def analyze_log_file():
    """Analyze wave parameters from log file"""
    print("\n" + "="*70)
    print("WAVE PARAMETERS FROM SIMULATION LOG")
    print("="*70)
    
    log_file = '/home/franek/TCLB/output/01clean_canal_Log_P00_00000000.csv'
    df = pd.read_csv(log_file)
    
    # Get wave parameters (they should be constant in the current setup)
    wave_params = df[['Wave_A', 'Wave_k_real', 'Wave_k_imag', 'Wave_w', 
                      'Wave_Phase', 'Wave_Period', 'Wave_Length']].iloc[-1]
    
    print("\nCurrent XML Parameters (from log):")
    for param, value in wave_params.items():
        print(f"  {param}: {value:.8f}")
    
    # Verify consistency
    wavelength = wave_params['Wave_Length']
    k_real = wave_params['Wave_k_real']
    expected_k = 2 * np.pi / wavelength
    
    print(f"\nParameter Consistency Check:")
    print(f"  Wavelength λ: {wavelength:.2f} LU")
    print(f"  k_real from XML: {k_real:.8f}")
    print(f"  Expected k = 2π/λ: {expected_k:.8f}")
    print(f"  Match: {np.isclose(k_real, expected_k)}")
    
    # Wave error tracking
    print(f"\nWave Error Tracking (Reflection Zone):")
    wave_error = df['WaveErrorInObj-Reflection_Zone'].values
    print(f"  Min: {wave_error.min():.8f}")
    print(f"  Max: {wave_error.max():.8f}")
    print(f"  Mean: {wave_error.mean():.8f}")
    print(f"  Std: {wave_error.std():.8f}")
    
    return dict(wave_params)

def compute_spatial_parameters():
    """Compute spatial wave parameters from dispersive wave theory"""
    print("\n" + "="*70)
    print("THEORETICAL WAVE PARAMETERS (Linear Shallow Water Theory)")
    print("="*70)
    
    # Physical parameters from XML
    Gravity = 0.0005  # g in LB units
    Height = 40.0     # h in LB units
    
    # Wave parameters from inlet analysis
    inlet_params = analyze_inlet_wave()
    omega = inlet_params['w']  # Angular frequency
    
    # Shallow water dispersion: ω² = gk*tanh(k*h)
    # For shallow water (k*h << 1): ω² ≈ gk*h => ω = √(gh)*k => c = √(gh)
    # So: k = ω / c = ω / √(gh)
    
    c_phase = np.sqrt(Gravity * Height)  # Wave speed
    k_from_theory = omega / c_phase
    lambda_from_theory = 2 * np.pi / k_from_theory
    
    print(f"\nShallow Water Wave Theory (h << λ):")
    print(f"  Gravity (g): {Gravity:.8f} [LB units]")
    print(f"  Water depth (h): {Height:.2f} [LU]")
    print(f"  Angular frequency (ω): {omega:.8f} [1/LU]")
    print(f"  Phase speed (c = √(gh)): {c_phase:.8f} [LU/iter]")
    print(f"  Computed k_real: {k_from_theory:.8f} [1/LU]")
    print(f"  Computed wavelength: {lambda_from_theory:.2f} [LU]")
    
    # Damping (no damping in ideal shallow water)
    k_imag = 0.0
    print(f"  Computed k_imag: {k_imag:.8f} (no damping in theory)")
    
    return {
        'k_real': k_from_theory,
        'wavelength': lambda_from_theory,
        'k_imag': k_imag,
        'phase_speed': c_phase
    }

def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "WAVE PARAMETER ANALYSIS" + " "*30 + "║")
    print("╚" + "="*68 + "╝")
    
    inlet_params = analyze_inlet_wave()
    log_params = analyze_log_file()
    theory_params = compute_spatial_parameters()
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    print("\nCurrent Configuration:")
    print(f"  XML Wave_A: {log_params['Wave_A']:.6f}")
    print(f"  XML Wave_k_real: {log_params['Wave_k_real']:.6f}")
    print(f"  XML Wave_k_imag: {log_params['Wave_k_imag']:.6f}")
    print(f"  XML Wave_w: {log_params['Wave_w']:.6f}")
    print(f"  XML Wave_Phase: {log_params['Wave_Phase']:.6f}")
    print(f"  XML Wave_Length: {log_params['Wave_Length']:.1f}")
    
    print("\nTheoretical Values (from Shallow Water Theory):")
    print(f"  k_real (theoretical): {theory_params['k_real']:.6f}")
    print(f"  Wavelength (theoretical): {theory_params['wavelength']:.1f}")
    print(f"  k_imag: {theory_params['k_imag']:.6f}")
    
    print("\nValidation:")
    k_match = np.isclose(log_params['Wave_k_real'], theory_params['k_real'], rtol=0.01)
    lambda_match = np.isclose(log_params['Wave_Length'], theory_params['wavelength'], rtol=0.01)
    
    if k_match and lambda_match:
        print("  ✓ XML parameters match theoretical prediction!")
        print("  ✓ Wave parameters are consistent with the system physics")
    else:
        print("  ✗ XML parameters differ from theory - may need adjustment")
        print(f"    - k_real difference: {abs(log_params['Wave_k_real'] - theory_params['k_real']):.8f}")
        print(f"    - Wavelength difference: {abs(log_params['Wave_Length'] - theory_params['wavelength']):.2f}")
    
    print("\nRecommendation:")
    print("  Use the computed theoretical parameters for accurate wave simulation:")
    print(f'  <Param name="Wave_k_real" value="{theory_params["k_real"]:.6f}"/>')
    print(f'  <Param name="Wave_k_imag" value="{theory_params["k_imag"]:.6f}"/>')
    print(f'  <Param name="Wave_Length" value="{theory_params["wavelength"]:.1f}"/>')

if __name__ == '__main__':
    main()
