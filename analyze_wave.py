#!/usr/bin/env python3
"""
Analyze wave parameters from VTK simulation output
Computes: k_real, k_imag, phase, frequency, amplitude, wavelength
"""

import numpy as np
import os
import glob
from scipy import signal
from scipy.fft import fft, fftfreq
import xml.etree.ElementTree as ET
import base64
import struct

def read_vti_file(filepath):
    """Read VTI (VTK Image Data) XML file with base64-encoded data"""
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Find ImageData element
    imgdata = root.find('ImageData')
    if imgdata is None:
        return None
    
    # Parse attributes
    extent = list(map(int, imgdata.get('WholeExtent', '0 0 0 0 0 0').split()))
    origin = list(map(float, imgdata.get('Origin', '0 0 0').split()))
    spacing = list(map(float, imgdata.get('Spacing', '1 1 1').split()))
    
    # Dimensions from extent: (x_max-x_min+1, y_max-y_min+1, z_max-z_min+1)
    nx = extent[1] - extent[0] + 1
    ny = extent[3] - extent[2] + 1
    nz = extent[5] - extent[4] + 1
    
    # Find Rho data
    piece = imgdata.find('Piece')
    if piece is None:
        return None
    
    celldata = piece.find('CellData')
    if celldata is None:
        return None
    
    rho_array = None
    for dataarray in celldata.findall('DataArray'):
        if dataarray.get('Name') == 'Rho':
            rho_array = dataarray
            break
    
    if rho_array is None:
        return None
    
    # Decode base64 data - keep all whitespace and newlines
    encoded_data = ''
    if rho_array.text:
        # Remove only leading/trailing whitespace, but keep internal newlines
        encoded_data = rho_array.text.strip()
        # Remove all internal whitespace (spaces, newlines, tabs)
        encoded_data = ''.join(encoded_data.split())
    
    if not encoded_data:
        return None
    
    decoded_data = base64.b64decode(encoded_data)
    
    # Interpret as Float64 (double)
    num_values = len(decoded_data) // 8
    if num_values == 0:
        return None
        
    values = struct.unpack(f'{num_values}d', decoded_data)
    data = np.array(values)
    
    # Reshape: note that VTK stores in Fortran order (z, y, x changes fastest)
    # But for 2D case with nz=1, we get (x*y*z,) which reshapes to (nz, ny, nx)
    data = data.reshape((nz, ny, nx), order='F')
    
    return data, (nx, ny, nz), origin, spacing

def analyze_wave_at_y(data, y_index, nx, k_range=(0.001, 0.05)):
    """Extract wave profile at constant y and analyze spatial frequency"""
    # Extract 1D profile
    profile = data[0, y_index, :]  # z=0 (2D), y=y_index, x=all
    
    if profile.size < 10:
        return None
    
    # Remove DC offset
    profile_ac = profile - np.mean(profile)
    
    # FFT to find dominant wavenumber
    X_freq = fftfreq(len(profile_ac), d=1.0)  # d=1.0 for lattice units
    X_fft = np.abs(fft(profile_ac))
    
    # Look for dominant frequency in positive range
    pos_freqs = X_freq[:len(X_freq)//2]
    pos_fft = X_fft[:len(X_fft)//2]
    
    # Filter to k_range
    mask = (pos_freqs > k_range[0]) & (pos_freqs < k_range[1])
    if mask.sum() == 0:
        print(f"No significant energy in k_range {k_range}")
        return None
    
    k_idx = np.argmax(pos_fft[mask]) + np.where(mask)[0][0]
    k_real = pos_freqs[k_idx]
    
    # Amplitude from FFT magnitude (normalized)
    amplitude = 2 * pos_fft[k_idx] / len(profile_ac)
    
    # Wavelength
    wavelength = 2 * np.pi / k_real if k_real > 0 else np.inf
    
    # Extract phase by fitting cosine
    # profile ≈ A * cos(k*x + phase)
    x = np.arange(len(profile_ac))
    
    # Fit: find phase by correlation with cos(k*x)
    phase_test = np.linspace(-np.pi, np.pi, 100)
    correlations = []
    for phase in phase_test:
        test_wave = amplitude * np.cos(k_real * x + phase)
        corr = np.correlate(profile_ac, test_wave, mode='same').max()
        correlations.append(corr)
    
    phase = phase_test[np.argmax(correlations)]
    
    return {
        'k_real': k_real,
        'wavelength': wavelength,
        'amplitude': amplitude,
        'phase': phase,
        'k_real_2pi': k_real * 2 * np.pi
    }

def main():
    # Find all VTI files
    output_dir = '/home/franek/TCLB/output'
    vti_files = sorted(glob.glob(os.path.join(output_dir, '*_P00_*.vti')))
    
    if not vti_files:
        print(f"No VTI files found in {output_dir}")
        return
    
    print(f"Found {len(vti_files)} VTI files")
    print(f"Analyzing first few timesteps...\n")
    
    # Analyze multiple timesteps
    results = []
    
    for vti_file in vti_files[5:15]:  # Analyze middle timesteps (after wave settles)
        try:
            basename = os.path.basename(vti_file)
            # Extract iteration from format: 01clean_canal_VTK_P00_00003000.vti
            iteration = int(basename.split('_')[-1].split('.')[0])
            data, dims, origin, spacing = read_vti_file(vti_file)
            
            if data is None:
                continue
            
            # Analyze at middle y position
            y_idx = dims[1] // 2
            
            wave_params = analyze_wave_at_y(data, y_idx, dims[0])
            
            if wave_params:
                results.append({
                    'iteration': iteration,
                    'file': os.path.basename(vti_file),
                    **wave_params
                })
                print(f"Iteration {iteration}:")
                print(f"  k_real = {wave_params['k_real']:.6f} [1/LU]")
                print(f"  Wavelength = {wave_params['wavelength']:.2f} [LU]")
                print(f"  Amplitude = {wave_params['amplitude']:.6f}")
                print(f"  Phase = {wave_params['phase']:.6f} [rad]")
                print()
        
        except Exception as e:
            print(f"Error processing {vti_file}: {e}")
            continue
    
    # Average results
    if results:
        print("\n" + "="*60)
        print("AVERAGED WAVE PARAMETERS:")
        print("="*60)
        
        avg_k_real = np.mean([r['k_real'] for r in results])
        avg_wavelength = np.mean([r['wavelength'] for r in results])
        avg_amplitude = np.mean([r['amplitude'] for r in results])
        avg_phase = np.mean([r['phase'] for r in results])
        
        print(f"k_real (2π/λ) = {avg_k_real:.6f} [1/LU]")
        print(f"Wavelength = {avg_wavelength:.2f} [LU]")
        print(f"Amplitude = {avg_amplitude:.6f}")
        print(f"Phase = {avg_phase:.6f} [rad]")
        print(f"\nFor XML file:")
        print(f'<Param name="Wave_k_real" value="{avg_k_real:.6f}"/>')
        print(f'<Param name="Wave_Length" value="{avg_wavelength:.1f}"/>')
        print(f'<Param name="Wave_A" value="{avg_amplitude:.6f}"/>')
        print(f'<Param name="Wave_Phase" value="{avg_phase:.6f}"/>')

if __name__ == '__main__':
    main()
