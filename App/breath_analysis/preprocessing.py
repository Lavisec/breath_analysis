import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

from constants import LOW_PASS_CUTOFF, WINDOW_NUM, RESAMPLED_FREQ, MANUAL_BASELINE

def get_raw_data(csv_file):
    """
    Load pressure data from a CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
        
    Returns:
    --------
    time : array-like
        Time axis from the CSV file
    pressures : dict
        Dictionary mapping each pressure column name (e.g. 'pressure1', 'pressure2')
        to its data array. Columns are detected automatically.
    sampling_rate : float
        Sampling rate in Hz (samples per second)
    """
    df = pd.read_csv(csv_file)
    
    df = df.iloc[:80000, :]
    
    # Calculate sampling rate from time data
    # min_dt = df['time'].diff().min() # fix here
    min_dt = 1/40
    sampling_rate = 1/min_dt
    time = np.arange(df['time'].iloc[0], df['time'].iloc[-1], min_dt)

    # Detect all pressure columns (columns whose stripped name matches 'pressure<digits>')
    pressure_cols = sorted(
        [c for c in df.columns if c.strip().lower().startswith('pressure') and c.strip()[8:].isdigit()],
        key=lambda c: int(c.strip()[8:])
    )

    pressures = {col.strip(): df[col].values for col in pressure_cols}

    results = []

    for col in pressure_cols:
        pressure = np.interp(time, df['time'].values, pressures[col.strip()])
        result = {
            'filename': os.path.basename(csv_file),
            'time': time,
            'samp_rate': sampling_rate,
            'pressure_raw': pressure
        }
        results.append(result)

    return results

def low_pass_filter(result, cutoff_freq=LOW_PASS_CUTOFF):
    """
    Apply a low-pass Butterworth filter to pressure data.
    
    Parameters:
    -----------
    result : dict
        Dictionary containing pressure data and sampling rate
    cutoff_freq : float
        Cutoff frequency in Hz for the low-pass filter. Default is LOW_PASS_CUTOFF (2 Hz).
        Cutoff frequency in Hz
        
    Returns:
    --------
    pressure_filtered : array-like
        Filtered pressure data
    """
    sampling_rate = result['samp_rate']
    pressure_data = result['pressure_raw']
    nyquist_freq = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    if normalized_cutoff >= 1:
        print(f"  Warning: Cutoff frequency {cutoff_freq} Hz exceeds Nyquist frequency {nyquist_freq:.2f} Hz")
        normalized_cutoff = 0.99
    
    b, a = butter(4, normalized_cutoff, btype='low')
    pressure_filtered = filtfilt(b, a, pressure_data)
    result['pressure_filtered'] = pressure_filtered

    return result

def get_baselines(result, n=WINDOW_NUM):
    """
    Calculate multiple baselines from pressure data by creating 50% overlapping windows
    of the data according to the number of windows (n) and returning each window's baseline.
    
    Parameters:
    -----------
    n : int
        Number of windows to create
    result : dict
        Dictionary containing pressure data and sampling rate
        
    Returns:
    --------
    baselines : list
        List of baseline values, one for each window
    """
    pressure = -result['pressure_filtered']
    pressure_len = len(pressure)
    all_baselines = []
    
    for j in range(n):
        window_num = 2**j  # Exponential number of windows
        # Calculate window size to fit all windows with 50% overlap spanning the entire data
        # With 50% overlap: total_span = window_size + (window_size/2) * (num_windows - 1)
        # Solving for window_size: window_size * (num_windows + 1) / 2 = pressure_len
        window_size = int(pressure_len * 2 / (window_num + 1))
        step_size = window_size // 2  # 50% overlap means step is half the window size
        
        baselines = []
        
        for i in range(window_num):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            # Ensure we don't exceed the data bounds
            if start_idx >= pressure_len:
                break
            if end_idx > pressure_len:
                end_idx = pressure_len
            
            # Extract window and calculate baseline
            window_data = pressure[start_idx:end_idx]
            
            if len(window_data) > 0:
                # Create histogram for this window with 1000 bins
                counts, bins = np.histogram(window_data, bins=100)
                max_index = counts.argmax()
                counts, bins = np.histogram(window_data, bins=10000, range=(bins[max_index],
                                                                            bins[max_index + 1]))
                bins = np.mean([bins[:-1], bins[1:]], axis=0)
                # plt.xlabel('Pressure', fontsize=11)
                # plt.ylabel('Frequency', fontsize=11)
                # plt.title(f'Pressure Distribution Histogram - Window {i+1}/{n}', fontsize=12)
                # plt.grid(True, alpha=0.3)
                # plt.show()
                
                # Find the maximum of the histogram
                max_index = counts.argmax()
                baseline = bins[max_index]
                baselines.append(baseline)
        
        all_baselines.append(baselines)
    result['baselines'] = all_baselines
    
    return result

def get_weighted_average_baseline(result):

    """
    Convert baseline values (one per window) into full baseline curves for the entire dataset.
    For each baseline set, finds window midpoints, interpolates linearly between them,
    and extrapolates to the beginning and end using first and last segment slopes.
    
    Parameters:
    -----------
    result : dict
        Dictionary containing baseline values and data length
        - 'baselines' : list of lists
            List of baseline values for each window configuration. Each inner list contains
            one baseline value per window.
        - 'data_length' : int
            Length of the dataset (number of samples)
    
    Returns:
    --------
    baseline : array
        Full baseline curve for the dataset, with length equal to data_length. This
        baseline is the weighted arithmetic mean of the window baselines, 
        interpolated and extrapolated to cover the entire dataset.
    """
    
    baselines = []
    baselines_weights = []
    total_weight = 0
    
    baselines_vals = result['baselines']
    data_length = len(result['pressure_filtered'])
    
    for baseline_idx, baseline_list in enumerate(baselines_vals):
        # Number of windows for this baseline set
        window_num = len(baseline_list)

        total_weight += window_num
        baselines_weights.insert(0, window_num)  # Insert at beginning to have weights in order of increasing windows
        
        # Calculate window positions - find middle indices of each window
        window_size = int(data_length * 2 / (window_num + 1))
        step_size = window_size // 2
        
        window_middle_indices = []
        for i in range(window_num):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            if end_idx > data_length:
                end_idx = data_length
            middle_idx = (start_idx + end_idx) // 2
            window_middle_indices.append(middle_idx)
        
        # Create full baseline curve with interpolation
        baseline_curve = np.zeros(data_length)
        
        # Calculate first segment slope (for extrapolation at beginning)
        if window_num > 1:
            first_slope = (baseline_list[1] - baseline_list[0]) / (window_middle_indices[1] - window_middle_indices[0])
        else:
            first_slope = 0
        
        # Calculate last segment slope (for extrapolation at end)
        if window_num > 1:
            last_slope = (baseline_list[-1] - baseline_list[-2]) / (window_middle_indices[-1] - window_middle_indices[-2])
        else:
            last_slope = 0
        
        # Fill the beginning with extrapolation using first slope
        for idx in range(window_middle_indices[0]):
            baseline_curve[idx] = baseline_list[0] - first_slope * (window_middle_indices[0] - idx)
        
        # Interpolate between window midpoints
        for i in range(len(window_middle_indices) - 1):
            start_idx = window_middle_indices[i]
            end_idx = window_middle_indices[i + 1]
            start_val = baseline_list[i]
            end_val = baseline_list[i + 1]
            
            # Linear interpolation
            indices = np.arange(start_idx, end_idx + 1)
            interpolated = start_val + (end_val - start_val) * (indices - start_idx) / (end_idx - start_idx)
            baseline_curve[start_idx:end_idx + 1] = interpolated
        
        # Fill the end with extrapolation using last slope
        for idx in range(window_middle_indices[-1] + 1, data_length):
            baseline_curve[idx] = baseline_list[-1] + last_slope * (idx - window_middle_indices[-1])
        
        baselines.append(baseline_curve)

    # Calculate weighted average baseline
    baseline=np.zeros(data_length)
    for i in range(len(baselines)):
        baseline += baselines[i] * (baselines_weights[i] / total_weight)
        
    baseline += MANUAL_BASELINE
    
    result['baseline'] = baseline
    
    result['pressure_bc'] = result['pressure_filtered'] - result['baseline']

    return result

def upsample_data(result, resampled_freq=RESAMPLED_FREQ):
    time = result['time']
    time_resampled = np.arange(time[0], time[-1], 1/resampled_freq)
    result['time_resampled'] = time_resampled
    result['resampled_freq'] = resampled_freq

    # Upsample to improve frequency resolution
    pressure_resampled = np.interp(time_resampled, time, result['pressure_bc'])

    nyquist_freq = resampled_freq / 2
    normalized_cutoff = LOW_PASS_CUTOFF / nyquist_freq

    if normalized_cutoff >= 1:
        print(f"  Warning: Cutoff frequency {LOW_PASS_CUTOFF} Hz exceeds Nyquist frequency {nyquist_freq:.2f} Hz")
        normalized_cutoff = 0.99
    
    b, a = butter(4, normalized_cutoff, btype='low')
    pressure_resampled = filtfilt(b, a, pressure_resampled)

    result['pressure_resampled'] = pressure_resampled
    
    return result
