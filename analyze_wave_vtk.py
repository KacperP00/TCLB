#!/usr/bin/env python3
"""
Analyze wave parameters from VTK simulation output using VTK library
Computes: k_real, k_imag, phase, frequency, amplitude, wavelength
"""

import numpy as np
import os
import glob
from scipy.fft import fft, fftfreq
import vtk

def read_vti_file(filepath):
    """Read VTI file using VTK library"""
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    
    image = reader.GetOutput()
    
    # Get dimensions
    nx, ny, nz = image.GetDimensions()
    
    # Get Rho (density) data
    celldata = image.GetCellData()
    rho_array = celldata.GetArray('Rho')
    
    if rho_array is None:
        return None
    
    # Convert to numpy array
    num_cells = image.GetNumberOfCells()
    data_np = np.zeros(num_cells)
    for i in range(num_cells):
        data_np[i] = rho_array.GetValue(i)
    
    # Reshape data
    # For 2D data with nz=1: reshape to (nx*ny, ) -> (ny, nx)
    # For 3D data: reshape to (nz, ny, nx)
    
    # Compute cell dimensions
    cell_nx = nx - 1
    cell_ny = ny - 1
    cell_nz = nz - 1
    
    if cell_nz == 0:
        cell_nz = 1  # 2D case
    
    data = data_np.reshape((cell_nz, cell_ny, cell_nx), order='F')
    
    return data, (cell_nx, cell_ny, cell_nz), image.GetOrigin(), image.GetSpacing()

def analyze_wave_at_y(data, y_index, nx, k_range=(0.001, 0.05)):
    """Extract wave profile at constant y and analyze spatial frequency"""
    # Extract 1D profile at z=0, y=y_index, all x
    profile = data[0, y_index, :] if data.shape[0] > 0 else None
    
    if profile is None or profile.size < 10:
        return None
    
    # Remove DC offset and normalize
    profile_ac = profile - np.mean(profile)
    
    if np.std(profile_ac) < 1e-10:
        return None  # No signal
    
    profile_ac = profile_ac / np.std(profile_ac)
    
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
    
    if k_real == 0:
        return None
    
    # Amplitude from original data
    amplitude = np.std(profile)
    
    # Wavelength
    wavelength = 2 * np.pi / k_real
    
    # Extract phase by fitting cosine
    x = np.arange(len(profile_ac))
    
    # Fit: find phase by correlation with cos(k*x)
    phase_test = np.linspace(-np.pi, np.pi, 100)
    max_corr = 0
    best_phase = 0
    
    for phase in phase_test:
        test_wave = np.cos(k_real * x + phase)
        corr = np.abs(np.correlate(profile_ac, test_wave, mode='same').max())
        if corr > max_corr:
            max_corr = corr
            best_phase = phase
    
    phase = best_phase
    
    return {
        'k_real': k_real,
        'wavelength': wavelength,
        'amplitude': amplitude,
        'phase': phase,
        'energy': np.sum(profile_ac**2)
    }

def main():
    # Find all VTI files
    output_dir = '/home/franek/TCLB/output'
    vti_files = sorted(glob.glob(os.path.join(output_dir, '*_P00_*.vti')))
    
    if not vti_files:
        print(f"No VTI files found in {output_dir}")
        return
    
    print(f"Found {len(vti_files)} VTI files")
    print(f"Analyzing timesteps...\n")
    
    # Analyze multiple timesteps
    results = []
    
    # Analyze timesteps from iteration 10000 onwards (after wave settles)
    stride = 5
    for vti_file in vti_files[::stride]:
        try:
            basename = os.path.basename(vti_file)
            iteration = int(basename.split('_')[-1].split('.')[0])
            
            data, dims, origin, spacing = read_vti_file(vti_file)
            
            if data is None:
                print(f"Failed to read {basename}")
                continue
            
            # Analyze at middle y position
            y_idx = min(16, dims[1] // 2)  # Use y=16 (middle of domain height)
            
            wave_params = analyze_wave_at_y(data, y_idx, dims[0])
            
            if wave_params and wave_params['amplitude'] > 0.0001:
                results.append({
                    'iteration': iteration,
                    'file': basename,
                    **wave_params
                })
                print(f"Iteration {iteration}:")
                print(f"  k_real = {wave_params['k_real']:.6f} [1/LU]")
                print(f"  Wavelength = {wave_params['wavelength']:.2f} [LU]")
                print(f"  Amplitude = {wave_params['amplitude']:.6f}")
                print(f"  Phase = {wave_params['phase']:.6f} [rad]")
                print(f"  Energy = {wave_params['energy']:.2f}")
                print()
        
        except Exception as e:
            basename = os.path.basename(vti_file)
            print(f"Error processing {basename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Average results
    if results:
        print("\n" + "="*60)
        print("AVERAGED WAVE PARAMETERS:")
        print("="*60)
        
        # Filter by energy to get the best results
        energies = [r['energy'] for r in results]
        median_energy = np.median(energies)
        good_results = [r for r in results if r['energy'] > median_energy * 0.8]
        
        if not good_results:
            good_results = results[-5:]  # Use last 5 if filtering doesn't work
        
        avg_k_real = np.mean([r['k_real'] for r in good_results])
        avg_wavelength = np.mean([r['wavelength'] for r in good_results])
        avg_amplitude = np.mean([r['amplitude'] for r in good_results])
        avg_phase = np.mean([r['phase'] for r in good_results])
        
        print(f"k_real (2π/λ) = {avg_k_real:.6f} [1/LU]")
        print(f"Wavelength = {avg_wavelength:.2f} [LU]")
        print(f"Amplitude = {avg_amplitude:.6f}")
        print(f"Phase = {avg_phase:.6f} [rad]")
        print(f"\nFor XML file:")
        print(f'<Param name="Wave_k_real" value="{avg_k_real:.6f}"/>')
        print(f'<Param name="Wave_Length" value="{avg_wavelength:.1f}"/>')
        print(f'<Param name="Wave_A" value="{avg_amplitude:.6f}"/>')
        print(f'<Param name="Wave_Phase" value="{avg_phase:.6f}"/>')
    else:
        print("No valid results found")

if __name__ == '__main__':
    main()
