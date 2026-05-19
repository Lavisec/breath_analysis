"""
Lavi's Baseline Analysis Module
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np
import scipy.signal as signal
from scipy.fft import fft, ifft, fftfreq
from scipy import stats

LOW_PASS_CUTOFF = 2  # Hz, cutoff frequency for low-pass filter
WINDOW_NUM = 5

def analysis():
    """
    Analyze all CSV files in the data directory by loading raw data,
    applying a low-pass filter, and plotting both on the same figure.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data/target')
    
    # Get all CSV files in the data directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in data directory.")
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        print(f"\nProcessing: {csv_file}")
        
        # Load raw data
        time, pressures_raw, sampling_rate = get_raw_data(file_path)
        print(f"  Sampling rate: {sampling_rate:.2f} Hz")
        print(f"  Applied low-pass filter with cutoff frequency: {LOW_PASS_CUTOFF} Hz")

        pressures_filtered = {}
        baselines = {}
        for col, pressure in pressures_raw.items():
            pressures_filtered[col] = low_pass_filter(pressure, sampling_rate)
            baselines_vals = get_baselines(WINDOW_NUM, pressures_filtered[col])
            baselines[col] = get_weighted_average_baseline(baselines_vals, len(pressures_filtered[col]))

        # Plot raw vs filtered data
        plot_data(time, pressures_raw, pressures_filtered, baselines, csv_file)

def analyze_file(file_path, test_flag=False):
    """
    Run the full analysis pipeline on a single CSV file and return the results.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file to analyze.

    Returns:
    --------
    dict with keys:
        filename   : str
        time       : array-like
        pressures_raw      : dict mapping column name -> raw pressure array
        pressures_filtered : dict mapping column name -> filtered pressure array
        baselines          : dict mapping column name -> baseline curve array
    """
    results = get_raw_data(file_path)
    

    if test_flag:
        result = results[0]
        result = low_pass_filter(result)
        result = get_baselines(result)
        result = get_weighted_average_baseline(result)
        result = upsample_data(result)

        result = analyze_breath(result)

        return result

    else:
        for result in results:
            result = low_pass_filter(result)
            result = get_baselines(result)
            result = get_weighted_average_baseline(result)
            result = upsample_data(result)

            result = analyze_breath(result)

        export_results(results)

        return results


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
    
    # Calculate sampling rate from time data
    min_dt = df['time'].diff().min()
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

def low_pass_filter_second_time(result, cutoff_freq=LOW_PASS_CUTOFF):
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
    sampling_rate = result['resampled_freq']
    pressure_data = result['pressure_resampled']
    nyquist_freq = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    if normalized_cutoff >= 1:
        print(f"  Warning: Cutoff frequency {cutoff_freq} Hz exceeds Nyquist frequency {nyquist_freq:.2f} Hz")
        normalized_cutoff = 0.99
    
    b, a = butter(4, normalized_cutoff, btype='low')
    pressure_filtered = filtfilt(b, a, pressure_data)
    result['pressure_resampled'] = pressure_filtered

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
    
    result['baseline'] = baseline
    
    result['pressure_bc'] = result['pressure_filtered'] - baseline

    return result

def upsample_data(result):
    resampled_freq = 1000
    time = result['time']
    time_resampled = np.arange(time[0], time[-1], 1/resampled_freq)
    result['time_resampled'] = time_resampled
    result['resampled_freq'] = resampled_freq

    # Upsample to improve frequency resolution
    pressure_resampled = np.interp(time_resampled, time, result['pressure_bc'])
    result['pressure_resampled'] = pressure_resampled
    
    return result

def analyze_breath(result):
    """
    Placeholder for any additional breath analysis steps that may be needed before thresholding.
    Currently does nothing but
    """
    result = low_pass_filter_second_time(result)
    
    inhale_parameters = {}
    
    inhale_parameters = find_amp_se_thresholds(result, inhale_parameters, type_flag='inhale')
    inhale_parameters = find_se_points(result, inhale_parameters, type_flag='inhale')
    inhale_parameters = find_peaks_durations(result, inhale_parameters, type_flag='inhale')
    inhale_parameters = remove_outliers(result, inhale_parameters)

    exhale_parameters = {}

    exhale_parameters = find_amp_se_thresholds(result, exhale_parameters, type_flag='exhale')
    exhale_parameters = find_se_points(result, exhale_parameters, type_flag='exhale')
    exhale_parameters = find_peaks_durations(result, exhale_parameters, type_flag='exhale')
    exhale_parameters = remove_outliers(result, exhale_parameters)

    exhale_parameters['parameters']['troughs'] = np.column_stack((exhale_parameters['parameters']['peaks'][:, 0],
                                                                   -exhale_parameters['parameters']['peaks'][:, 1]))
    del exhale_parameters['parameters']['peaks']
    exhale_parameters['threshold_dict']['amplitude_threshold'] = -exhale_parameters['threshold_dict']['amplitude_threshold']

    result['inhale_parameters'] = inhale_parameters
    result['exhale_parameters'] = exhale_parameters

    result = create_event_list(result)
    
    return result

def find_amp_se_thresholds(result, parameters, type_flag):

    
    if type_flag == 'inhale':
        pressure_resampled = result['pressure_resampled']
        pressure_bc = result['pressure_bc']
    else:    
        pressure_resampled = -result['pressure_resampled']
        pressure_bc = -result['pressure_bc']
    resampled_freq = result['resampled_freq']
    freq_bc = result['samp_rate']
    
    # Define a threshold for inhale detection
    positive_vals = pressure_resampled[pressure_resampled > 0]
    amplitude_threshold = np.median(positive_vals) / 5 # Code debt here
    parameters['threshold_dict'] = {'amplitude_threshold': amplitude_threshold}
    
    # Find points above the threshold
    target_points = np.where(pressure_resampled > amplitude_threshold)[0]

    # Find the most common gap between above-threshold points to set the split threshold
    target_diffs = np.diff(target_points)
    target_diffs_hist = np.histogram(target_diffs, bins=2000, range=(2, 2000)) # Code debt here
    most_common_diff = (target_diffs_hist[1][np.argmax(target_diffs_hist[0])]) / resampled_freq
    se_threshold_hist = most_common_diff / 3 # Code debt here, maybe use freq instead of hist
    parameters['threshold_dict']['se_threshold_hist'] = se_threshold_hist

    # Find the dominant inhale frequency using FFT
    n = len(pressure_bc)
    fft_vals = np.abs(np.fft.rfft(pressure_bc))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / freq_bc)

    # Exclude DC component (index 0)
    fft_vals[fft_freqs<0.001] = 0
    dominant_freq = fft_freqs[np.argmax(fft_vals)]
    target_period_samples = 1 / dominant_freq
    target_se_threshold_fft = target_period_samples / 3
    parameters['threshold_dict']['se_threshold_fft'] = target_se_threshold_fft
    
    return parameters

def find_se_points(result, parameters, type_flag):
    """
    Find inhale start and end points in the pressure data by defining a threshold for points
    to be considered as parts of an inhale, finding the ruling frequency of the signal and declaring
    it as the frequency of the inhale, finding the differences between every two consecutive points
    in the signal which are above the threshold, and declaring points with differences in the vicinity
    of the period corresponding to the inhale frequency as inhale start and end points.

    Parameters:
    -----------
    result : dict
        Dictionary containing 'time', 'pressure_filtered', and 'baseline' as returned by analyze_file().

    Returns:
    --------
    result : dict
        Updated result dictionary with 'inhale_se_points' added as an (N, 2) array of
        (start_idx, end_idx) pairs (indices into the resampled time axis) marking each inhale.
    """

    if type_flag == 'inhale':
        pressure_resampled = result['pressure_resampled']
    else:
        pressure_resampled = -result['pressure_resampled']
    amplitude_threshold = parameters['threshold_dict']['amplitude_threshold']
    resampled_freq = result['resampled_freq']
    se_threshold = parameters['threshold_dict']['se_threshold_fft'] * resampled_freq

    # Find points above the threshold
    target_points = np.where(pressure_resampled > amplitude_threshold)[0]
    if len(target_points) == 0:
        parameters['se_points'] = np.empty((0, 2), dtype=int)
        return parameters

    # Identify start and end indices of each inhale
    target_diffs = np.diff(target_points)
    start_idx = target_points[np.where(target_diffs > se_threshold)[0] + 1]
    start_idx = [target_points[0], *start_idx]

    # Adding the beginning of the first inhale and the end of the last inhale to the start and
    # end indices respectively.
    end_idx = target_points[np.where(target_diffs > se_threshold)[0]]
    end_idx = [*end_idx, target_points[-1]]

    se_points = np.column_stack((start_idx, end_idx))

    # Dealing with cases where the start and end points are the same (e.g. when there is only one point above the threshold)
    del_points = np.where((se_points[:, 1] - se_points[:, 0]) == 0)[0]
    se_points = np.delete(se_points, del_points, axis=0)

    parameters['se_points'] = se_points

    return parameters

def find_peaks_durations(result, parameters, type_flag):
    """
    Calculate inhale parameters such as duration and volume for each detected inhale.

    Parameters:
    -----------
    result : dict
        Dictionary containing 'time_resampled', 'pressure_filtered', and 'inhale_se_points'.

    Returns:
    --------
    result : dict
        Updated result dictionary with 'inhale_parameters' added as a list of dicts containing
        parameters for each inhale (e.g. duration, volume and peak).
    """
    if type_flag == 'inhale':
        pressure_resampled = result['pressure_resampled']
    else:
        pressure_resampled = -result['pressure_resampled']
    time_resampled = result['time_resampled']
    resampled_freq = result['resampled_freq']
    se_points = parameters['se_points']

    duration = np.zeros((len(se_points), 1))
    volume = np.zeros((len(se_points), 1))
    peak = np.zeros((len(se_points), 2))

    for i in range(len(se_points)):
        start_idx = se_points[i, 0]
        end_idx = se_points[i, 1]
        
        duration[i] = time_resampled[end_idx] - time_resampled[start_idx]
        volume[i] = np.trapezoid(pressure_resampled[start_idx:end_idx], dx=1/resampled_freq)
        peak_ind = start_idx + np.argmax(pressure_resampled[start_idx:end_idx])
        peak[i] = np.array([[peak_ind, pressure_resampled[peak_ind]]])
    
    

    parameters['parameters'] = {
        'duration': duration,
        'volume': volume,
        'peaks': peak
    }

    return parameters

def remove_outliers(result, parameters):
    
    peaks = parameters['parameters']['peaks']
    durations = parameters['parameters']['duration']

    duration_low_threshold = 0.15
    duration_high_threshold = 10

    parameters['threshold_dict']['duration_low_threshold'] = duration_low_threshold
    parameters['threshold_dict']['duration_high_threshold'] = duration_high_threshold

    # plt.hist(result['parameters']['duration'], bins=2000, color='blue', alpha=0.7, range=(0, 20))
    # plt.show()

    delete_points_durations = np.where((durations < duration_low_threshold) 
                                       | (durations > duration_high_threshold))[0]
    parameters['outliers'] = {'duration_outliers': delete_points_durations}

    parameters['parameters']['duration'] = np.delete(parameters['parameters']['duration'], delete_points_durations, axis=0)
    parameters['parameters']['volume'] = np.delete(parameters['parameters']['volume'], delete_points_durations, axis=0)
    parameters['parameters']['peaks'] = np.delete(parameters['parameters']['peaks'], delete_points_durations, axis=0)
    parameters['se_points'] = np.delete(parameters['se_points'], delete_points_durations, axis=0)
    
    peaks = parameters['parameters']['peaks']
    
    peak_values = peaks[:, 1]
    sort_order = np.argsort(peak_values)
    sorted_peak_values = peak_values[sort_order]

    z_peaks = np.append(stats.zscore(np.diff(np.log(sorted_peak_values))), 0)

    mid = len(sort_order) // 2

    first_z = z_peaks[:mid]
    last_z = z_peaks[mid:]
    
    first_z[0] = 18 # Code Debt
    last_z[-1] = 18
    
    first_pos = np.where(first_z > 2)[0][-1]
    last_pos = mid + np.where(last_z > 2)[0][0]

    mask = np.zeros(len(sort_order), dtype=bool)
    mask[first_pos:last_pos + 1] = True
    delete_points_peaks = sort_order[~mask]
    parameters['outliers']['peak_outliers'] = delete_points_peaks

    parameters['parameters']['duration'] = np.delete(parameters['parameters']['duration'], delete_points_peaks, axis=0)
    parameters['parameters']['volume'] = np.delete(parameters['parameters']['volume'], delete_points_peaks, axis=0)
    parameters['parameters']['peaks'] = np.delete(parameters['parameters']['peaks'], delete_points_peaks, axis=0)
    parameters['se_points'] = np.delete(parameters['se_points'], delete_points_peaks, axis=0)
    
    # plt.figure(figsize=(10, 6))
    # plt.subplot(1, 2, 1)
    # plt.hist(updated_peak, bins=2000, color='orange', alpha=0.7)
    # plt.title('Peak Values After Outlier Removal')
    # plt.xlabel('Peak Pressure (arbitrary units)')
    # plt.ylabel('Count')
    # plt.subplot(1, 2, 2)
    # plt.hist(result['parameters']['peaks'], bins=2000, color='orange', alpha=0.7)
    # plt.title('Peak Values Before Outlier Removal')
    # plt.xlabel('Peak Pressure (arbitrary units)')
    # plt.ylabel('Count')
    # plt.show()

    # plt.plot(result['pressure_filtered'], label='Resampled', color='blue', alpha=0.6, linewidth=1)
    # plt.plot(result['inhale_se_points'][:, 0], result['pressure_filtered'][result['inhale_se_points'][:, 0]]+0.5, color='red', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Inhale Start')
    # plt.plot(result['inhale_se_points'][:, 1], result['pressure_filtered'][result['inhale_se_points'][:, 1]]-0.5, color='green', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Inhale End')
    # plt.plot(result['inhale_se_points'][delete_points_peak, 0], result['pressure_filtered'][result['inhale_se_points'][delete_points_peak, 0]], color='black', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Peak Outlier Start')
    # plt.plot(result['inhale_se_points'][delete_points_peak, 1], result['pressure_filtered'][result['inhale_se_points'][delete_points_peak, 1]], color='black', alpha=0.6, linewidth=1, marker='x', linestyle='None', label='Peak Outlier End')
    # plt.legend()
    # plt.title('Inhale Parameters with Peak Outliers Marked')
    # plt.show()
    # plt.plot(result['pressure_filtered'], label='Resampled', color='blue', alpha=0.6, linewidth=1)
    # plt.plot(result['inhale_se_points'][:, 0], result['pressure_filtered'][result['inhale_se_points'][:, 0]]+0.5, color='red', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Inhale Start')
    # plt.plot(result['inhale_se_points'][:, 1], result['pressure_filtered'][result['inhale_se_points'][:, 1]]-0.5, color='green', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Inhale End')
    # plt.plot(result['inhale_se_points'][delete_points_duration, 0], result['pressure_filtered'][result['inhale_se_points'][delete_points_duration, 0]], color='black', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Faulty Start')
    # plt.plot(result['inhale_se_points'][delete_points_duration, 1], result['pressure_filtered'][result['inhale_se_points'][delete_points_duration, 1]], color='black', alpha=0.6, linewidth=1, marker='x', linestyle='None', label='Faulty End')
    # plt.legend()
    # plt.show()
    
    # print(duration)
    # plt.plot(result['pressure_filtered'], label='Resampled', color='blue', alpha=0.6, linewidth=1)
    # plt.plot(result['inhale_se_points'][:, 0], result['pressure_filtered'][result['inhale_se_points'][:, 0]]+0.5, color='red', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Inhale Start')
    # plt.plot(result['inhale_se_points'][:, 1], result['pressure_filtered'][result['inhale_se_points'][:, 1]]-0.5, color='green', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Inhale End')
    # plt.plot(faluty_points[:, 0], result['pressure_filtered'][faluty_points[:, 0]], color='black', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Faulty Start')
    # plt.plot(faluty_points[:, 1], result['pressure_filtered'][faluty_points[:, 1]], color='black', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Faulty End')
    # plt.legend()
    # plt.title('Inhale Parameters with Faulty Points')
    # plt.show()

    # pressure = result['pressure_filtered']
    # se_points = result['inhale_se_points']

    # parameters_list = []
    # for start_idx, end_idx in se_points:
    #     if end_idx <= start_idx:
    #         continue
    #     inhale_duration = time_resampled[end_idx] - time_resampled[start_idx]
    #     inhale_volume = np.trapezoid(pressure[start_idx:end_idx], dx=1/resampled_fs)
    #     inhale_peak = np.max(pressure[start_idx:end_idx])
    #     parameters_list.append({
    #         'duration': inhale_duration,
    #         'volume': inhale_volume,
    #         'peaks': inhale_peak
    #     })
    # result['inhale_parameters'] = parameters_list

    # plt.hist(result['inhale_se_points'][:, 1] - result['inhale_se_points'][:, 0], bins=1000, range=(-2000, 2000))
    # plt.show()

    # Plot inhale parameters for visualization
    # durations = [p['duration'] for p in result['inhale_parameters']]
    # volumes = [p['volume'] for p in result['inhale_parameters']]
    # peaks = [p['peaks'] for p in result['inhale_parameters']]

    # plt.hist(duration, bins=2000, color='blue', alpha=0.7, range=(0, 5))
    # plt.title('Inhale Durations')
    # plt.xlabel('Duration (s)')
    # plt.ylabel('Count')
    # plt.show()

    # plt.subplot(1, 4, 2)
    # plt.hist(volumes, bins=200, color='green', alpha=0.7, range=(0, 40))
    # plt.title('Inhale Volumes')
    # plt.xlabel('Volume (arbitrary units)')
    # plt.ylabel('Count')

    # plt.subplot(1, 4, 3)
    # plt.hist(peaks, bins=int(max(peaks))*10, color='orange', alpha=0.7, range=(0, max(peaks) * 1.1))
    # plt.title('Inhale Peaks')
    # plt.xlabel('Peak Pressure (arbitrary units)')
    # plt.ylabel('Count')

    # plt.subplot(1, 4, 4)
    # plt.plot(result['time_resampled'], result['pressure_filtered'], label='Resampled', color='blue', alpha=0.6, linewidth=1)
    # plt.title('Resampled Pressure')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Pressure (arbitrary units)')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    return parameters
    
def create_event_list(result):
    """
    Fill in later. 

    """

    peaks = result['inhale_parameters']['parameters']['peaks']
    troughs = result['exhale_parameters']['parameters']['troughs']
    pressure = result['pressure_resampled']
    freq = result['resampled_freq']

    extrimum_list = []
    event_list = []

    peaks_troughs = np.concatenate((peaks, troughs), axis=0)
    sorted_peaks_troughs = peaks_troughs[np.argsort(peaks_troughs[:, 0])]

    for ev in sorted_peaks_troughs:
        if ev[1] > 0:
            type = 'inhale'
            
            # Find negative values to the left
            left_start = 0 if int(ev[0]) < 30*freq else int(ev[0]) - int(30*freq)
            left_region = pressure[left_start:int(ev[0])]
            left_negatives = np.where(left_region < 0)[0]
            if len(left_negatives) == 0:
                lim_left = left_start
            else:
                lim_left = left_start + left_negatives[-1]
            
            # Find start point (positive values after left boundary)
            start_region = pressure[lim_left:int(ev[0])]
            start_positives = np.where(start_region > 0)[0]
            start = lim_left + start_positives[0] if len(start_positives) > 0 else lim_left
            
            # Find negative values to the right
            right_end = min(int(ev[0]) + int(30*freq), len(pressure))
            right_region = pressure[int(ev[0]):right_end]
            right_negatives = np.where(right_region < 0)[0]
            if len(right_negatives) == 0:
                lim_right = right_end
            else:
                lim_right = int(ev[0]) + right_negatives[0]
            
            # Find end point (positive values up to right boundary)
            end_region = pressure[int(ev[0]):lim_right]
            end_positives = np.where(end_region > 0)[0]
            end = int(ev[0]) + end_positives[-1] if len(end_positives) > 0 else int(ev[0])
        else:
            type = 'exhale'
            
            # Find positive values to the left
            left_start = 0 if int(ev[0]) < 30*freq else int(ev[0]) - int(30*freq)
            left_region = pressure[left_start:int(ev[0])]
            left_positives = np.where(left_region > 0)[0]
            if len(left_positives) == 0:
                lim_left = left_start
            else:
                lim_left = left_start + left_positives[-1]
            
            # Find start point (negative values after left boundary)
            start_region = pressure[lim_left:int(ev[0])]
            start_negatives = np.where(start_region < 0)[0]
            start = lim_left + start_negatives[0] if len(start_negatives) > 0 else lim_left
            
            # Find positive values to the right
            right_end = min(int(ev[0]) + int(30*freq), len(pressure))
            right_region = pressure[int(ev[0]):right_end]
            right_positives = np.where(right_region > 0)[0]
            if len(right_positives) == 0:
                lim_right = right_end
            else:
                lim_right = int(ev[0]) + right_positives[0]
            
            # Find end point (negative values up to right boundary)
            end_region = pressure[int(ev[0]):lim_right]
            end_negatives = np.where(end_region < 0)[0]
            end = int(ev[0]) + end_negatives[-1] if len(end_negatives) > 0 else int(ev[0])
        
        event = {
            'start': start,
            'end': end,
            'duration': (end - start) / freq,
            'type': type
        }
        extrimum_list.append(event)

    for ind in range(len(extrimum_list) - 1):
        event_list.append(extrimum_list[ind])
        event_list.append({
            'start': extrimum_list[ind]['end'],
            'end': extrimum_list[ind + 1]['start'],
            'duration': (extrimum_list[ind + 1]['start'] - extrimum_list[ind]['end']) / freq,
            'type': 'pause (' + extrimum_list[ind]['type'] + ' - ' + extrimum_list[ind + 1]['type'] + ')'
        })
    event_list.append(extrimum_list[-1])
    
    result['event_list'] = event_list
    result['extrimum_list'] = extrimum_list
    return result

def export_results(results):
    """
    Creates a new CSV file and exports the results of the breath analysis to it.
    The file is saved next to the source file with ' - analysis' appended to the name.
    Results are placed side by side: type - p1, start - p1, end - p1, duration - p1,
    type - p2, start - p2, ...

    Parameters:
    -----------
    results : list
        List of result dicts from the breath analysis pipeline. Each dict must contain
        'filename' and 'event_list'.
    """
    source_name = os.path.splitext(results[0]['filename'])[0]
    output_file = source_name + ' - analysis.csv'

    frames = []
    for ind, result in enumerate(results):
        label = f'p{ind + 1}'
        df = pd.DataFrame(result['event_list'])[['type', 'start', 'end', 'duration']]
        df.columns = [f'{col} - {label}' for col in df.columns]
        frames.append(df)

    combined = pd.concat(frames, axis=1)
    combined.to_csv(output_file, index=False)

def breath_matrix (result, parameters):
    breaths = np.zeros((len(results['inhale_parameters']['parameters']['peaks']), 3*results['resampled_freq'] + 1))
    for ind, peak_data in enumerate(results['inhale_parameters']['parameters']['peaks'][1:]):
            breaths[ind,:] = (results['pressure_resampled'][peak_data[0].astype(int) - 1500:peak_data[0].astype(int) + 1501]) / peak_data[1]
            
def plot_data(time, pressures_raw, pressures_filtered, baselines, csv_file):
    """
    Plot raw vs filtered pressure data for each pressure column with the weighted baseline curve.

    Parameters:
    -----------
    time : array-like
        Time axis
    pressures_raw : dict
        Dictionary mapping column name -> raw pressure array
    pressures_filtered : dict
        Dictionary mapping column name -> filtered pressure array
    baselines : dict
        Dictionary mapping column name -> baseline curve array
    csv_file : str
        Filename for the plot title
    """
    cols = list(pressures_raw.keys())
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(f'Raw vs Filtered Pressure Data - {csv_file}', fontsize=14, fontweight='bold')

    for ax, col in zip(axes, cols):
        ax.plot(time, pressures_raw[col], label=f'Raw {col}', color='blue', alpha=0.6, linewidth=1)
        ax.plot(time, pressures_filtered[col], label=f'Filtered {col}', color='darkblue', linewidth=1.5)
        ax.plot(time, baselines[col], label='Baseline', color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Pressure', fontsize=11)
        ax.set_title(f'{col}: Raw vs Filtered', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_threshold_outlier_sensitivity(file_path,
                                       amplitude_multipliers=None,
                                       inhale_end_start_multipliers=None,
                                       duration_low_threshold=0.15,
                                       duration_high_threshold=10):
    """
    Sweep over a grid of amplitude_threshold and inhale_end_start_threshold values and
    plot a 3D surface where the color (and z-axis) represents the number of inhales
    discarded by the low/high duration thresholds.

    The two threshold axes are expressed as multipliers of the auto-detected defaults so
    the grid is always centered on the algorithm's natural operating point.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to analyse.
    amplitude_multipliers : array-like, optional
        Multipliers applied to the auto-detected amplitude_threshold.
        Defaults to np.linspace(0.2, 3.0, 30).
    inhale_end_start_multipliers : array-like, optional
        Multipliers applied to the auto-detected inhale_end_start_threshold.
        Defaults to np.linspace(0.2, 3.0, 30).
    duration_low_threshold : float
        Minimum acceptable inhale duration (seconds). Default 0.15.
    duration_high_threshold : float
        Maximum acceptable inhale duration (seconds). Default 10.
    """
    if amplitude_multipliers is None:
        amplitude_multipliers = np.linspace(0.2, 3.0, 30)
    if inhale_end_start_multipliers is None:
        inhale_end_start_multipliers = np.linspace(0.2, 3.0, 30)

    amplitude_multipliers = np.asarray(amplitude_multipliers)
    inhale_end_start_multipliers = np.asarray(inhale_end_start_multipliers)

    # ------------------------------------------------------------------ #
    # 1. Preprocessing (identical to analyze_file up to resampling)
    # ------------------------------------------------------------------ #
    time, pressures_raw, sampling_rate = get_raw_data(file_path)

    pressure_raw = pressures_raw['pressure1']
    pressure_filtered = low_pass_filter(pressure_raw, sampling_rate)
    baselines_vals = get_baselines(WINDOW_NUM, pressure_filtered)
    baseline = get_weighted_average_baseline(baselines_vals, len(pressure_filtered))

    pressure = pressure_filtered - baseline

    time_resampled = np.linspace(time.iloc[0], time.iloc[-1], len(time) * 50)
    pressure_resampled = np.interp(time_resampled, time, pressure)
    resampled_fs = 1.0 / (time_resampled[1] - time_resampled[0])

    # ------------------------------------------------------------------ #
    # 2. Compute default (auto-detected) thresholds
    # ------------------------------------------------------------------ #
    positive_vals = pressure_resampled[pressure_resampled > 0]
    default_amplitude_threshold = np.median(positive_vals) / 9

    inhale_points_default = np.where(pressure_resampled > default_amplitude_threshold)[0]
    inhale_diffs_default = np.diff(inhale_points_default)
    hist_counts, hist_bins = np.histogram(inhale_diffs_default, bins=2000, range=(2, 2000))
    most_common_diff = hist_bins[np.argmax(hist_counts)]
    # Convert from index units to seconds by accessing time_resampled
    default_ies_idx = int(most_common_diff / 5)
    default_inhale_end_start_threshold_sec = time_resampled[default_ies_idx] - time_resampled[0]
    print(f"Auto-detected defaults: amplitude_threshold={default_amplitude_threshold:.4f}, "
          f"inhale_end_start_threshold={default_inhale_end_start_threshold_sec:.4f} s")

    amp_thresholds = default_amplitude_threshold * amplitude_multipliers
    ies_thresholds_sec = default_inhale_end_start_threshold_sec * inhale_end_start_multipliers

    # ------------------------------------------------------------------ #
    # 3. Sweep the grid
    # ------------------------------------------------------------------ #
    outlier_counts = np.zeros((len(amp_thresholds), len(ies_thresholds_sec)), dtype=int)

    for i, amp_thresh in enumerate(amp_thresholds):
        inhale_points = np.where(pressure_resampled > amp_thresh)[0]
        if len(inhale_points) < 2:
            outlier_counts[i, :] = 0
            continue

        inhale_diffs_sec = time_resampled[inhale_points[1:]] - time_resampled[inhale_points[:-1]]

        for j, ies_thresh in enumerate(ies_thresholds_sec):
            split_mask = inhale_diffs_sec > ies_thresh
            starts = inhale_points[np.concatenate(([True], split_mask))]
            ends   = inhale_points[np.concatenate((split_mask, [True]))]
            se = np.column_stack((starts, ends))

            # Remove zero-length segments
            se = se[se[:, 1] - se[:, 0] > 0]
            if len(se) == 0:
                outlier_counts[i, j] = 0
                continue

            durations = time_resampled[se[:, 1]] - time_resampled[se[:, 0]]
            n_outliers = int(np.sum(
                (durations < duration_low_threshold) | (durations > duration_high_threshold)
            ))
            outlier_counts[i, j] = n_outliers

    # ------------------------------------------------------------------ #
    # 4. 2-D heatmap plot
    # ------------------------------------------------------------------ #
    Z = np.log(outlier_counts.astype(float))

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        Z.T,
        origin='lower',
        aspect='auto',
        extent=[amplitude_multipliers[0], amplitude_multipliers[-1],
                inhale_end_start_multipliers[0], inhale_end_start_multipliers[-1]],
        cmap='viridis',
    )

    fig.colorbar(im, ax=ax, label='Number of outliers (duration)')

    ax.set_xlabel('Amplitude threshold multiplier')
    ax.set_ylabel('Inhale gap threshold multiplier (×{:.4f} s)'.format(default_inhale_end_start_threshold_sec))
    ax.set_title(
        f'Duration-outlier sensitivity\n'
        f'(duration low={duration_low_threshold}s, high={duration_high_threshold}s)\n'
        f'{os.path.basename(file_path)}',
        fontsize=11
    )

    # Mark the auto-detected defaults (multiplier = 1.0)
    ax.scatter(
        [1.0],
        [1.0],
        color='red', s=80, zorder=5, label='Auto-detected defaults (1.0×)'
    )
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = analyze_file("/home/aviv/Work/Anat's Lab/Breath Analysis/data/data_20230705_112943_SA032.csv")
    # plot_threshold_outlier_sensitivity("/home/aviv/Work/Anat's Lab/Breath Analysis/data/data_20221106_073425_AA013.csv")