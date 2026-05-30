import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import (INITIAL_UPSAMPLE, SECUNDO_CONST, DC_EXCLUSION_FREQ, TIME_THRESHOLD_DIVISOR,
                       WINDOW_NUM, BL_HIST_RES, AMP_DISCRATISATION)

def preprocess_file(data):
    """
    Run the preprocessing steps on a single CSV file and return the intermediate results.
    This includes loading the raw data, applying the low-pass filter, and calculating baselines.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file to preprocess.

    Returns:
    --------
    dict with keys:
        filename   : str
        time       : array-like
        pressures_raw      : dict mapping column name -> raw pressure array
    """

    for dataset in data:
        nl = find_noise_level(dataset)
        dataset['nl'] = nl
        dataset['pressure'], dataset['time'] = upsample(dataset['time_raw'], dataset['pressure_raw'], INITIAL_UPSAMPLE)
        dataset['samp_rate'] = INITIAL_UPSAMPLE
        dataset['baseline'] = WMD(dataset, {'window_length':1000, 'overlap':50},
                                  find_baseline, bl_parameters={'n':WINDOW_NUM })
        # dataset['baseline'] = find_baseline(dataset, {'n': 5})
        dataset['pressure_bc'] = dataset['pressure'] - dataset['baseline']
        dataset['inh_amp_th'] = WMD(dataset, {'window_length':1000, 'overlap':50}, find_peak_th, peak_th_parameters={})
        dataset['exh_amp_th'] = WMD(dataset, {'window_length':1000, 'overlap':50}, find_trough_th, trough_th_parameters={})
        # dataset['inh_amp_th'] = nl
        # dataset['exh_amp_th'] = -nl
        dataset['time_th'] = WMD(dataset, {'window_length':1000, 'overlap':50},
                                 find_time_th, time_th_parameters={})
        # dataset['time_th'] = find_time_th(dataset, {})
    
    return data

def read_data(file_path):
    """
    Read the raw data from the CSV file and return a dict with the name of the file, time and raw pressure arrays.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file to read.

    Returns:
    --------
    dict with keys:
        filename   : str
        time       : array-like
        pressures_raw      : dict mapping column name -> raw pressure array
    """
    df = pd.read_csv(file_path)
    
    # Calculate sampling rate from time data
    time = df['time'].values

    # Detect all pressure columns (columns whose stripped name matches 'pressure<digits>')
    pressure_cols = [c for c in df.columns if c.strip().lower().startswith('pressure')]

    pressures = {col.strip(): df[col].values for col in pressure_cols}

    data = []

    for col in pressure_cols:
        pressure = pressures[col.strip()]
        result = {
            'filename': os.path.basename(file_path),
            'time_raw': time,
            'pressure_raw': pressure
        }
        data.append(result)
    
    return data

def find_noise_level(dataset):
    """
    Analyze the raw pressure data to estimate the noise level, which can be used for setting filter parameters.

    Parameters:
    -----------
    dataset : dict
        A dict containing 'time' and 'pressures_raw' for a single pressure column.

    Returns:
    --------
    float
        Estimated noise level.
    """
    pressure = dataset['pressure_raw']

    noise_range = np.diff(np.unique(np.sort(pressure)))
    nl = np.min(noise_range) * SECUNDO_CONST

    return nl
    
def upsample(time, pressure, target_rate):
    """
    Upsample the raw pressure data to the target sampling rate.

    Parameters:
    -----------
    time : array-like
        The original time array.
    pressure : array-like
        The original pressure array.
    target_rate : float
        The target sampling rate in Hz to which the data should be resampled.

    Returns:
    --------
    dict
        Updated dataset with upsampled 'time' and 'pressures_raw'.
    """
    ret_time = np.arange(time[0], time[-1] + 1/target_rate, 1/target_rate)
    ret_pressure = np.interp(ret_time, time, pressure)
    return ret_pressure, ret_time

def WMD(dataset, WMD_parameters, func, **kwargs):
    """
    Apply the Window Me That (WMD) algorithm to the pressure data. This is a general function that can be used to apply
    any processing function (like baseline finding, amplitude thresholding, time thresholding) in a windowed manner. 
    
    The function takes the dataset, WMD parameters (a dict that contains the time length of the windows and the level
    of overlap for the windows), the processing function to apply, and any additional parameters required by that function. 
    It then applies the processing function to each window of the data and combines the results by sewing the results
    of the windows together to create a continuous output.

    The sewing process is as follows: For each window, we calculate the result using the provided function. We then
    take the mid-point of each window and assign the result of that window to that mid-point. After processing all windows,
    we have a set of results at the mid-points of the windows. We then use linear interpolation to fill in the values between
    the mid-points, creating a continuous output curve. For the beginning and end of the data, we can use extrapolation based
    on the slope of the first and last segments to fill in those values. 

    Parameters:
    -----------
    dataset : dict
        A dict containing 'time' and 'pressures_raw' for a single pressure column.
    WMD_parameters : dict
        Parameters required for the WMD algorithm:
            - window_length : int
                The length of each window in samples in seconds.
            - overlap : float
                The percentage of overlap between consecutive windows (0 to 100).
    func : callable
        The function to apply for WMD, which should take the dataset and parameters as input.
    **kwargs :
        Additional parameters required by the WMD function.
    """
    
    samp_rate = dataset['samp_rate']
    window_length = WMD_parameters.get('window_length', 1000) * samp_rate  # Default window length in samples
    overlap = WMD_parameters.get('overlap', 50)  # Default overlap percentage
    data_len = len(dataset['pressure'])

    start_ind = 0
    end_ind = window_length
    step_size = int(window_length * (1 - overlap / 100))  # Calculate step size based on overlap

    results = []
    mid_points = []

    if data_len < window_length:
        # If the data is shorter than the window length, just apply the function to the whole dataset
        result = func(dataset, **kwargs)
        return np.full(data_len, result)  # Return a constant array with the result

    while end_ind < data_len:
        window_dataset = {
            'time': dataset.get('time', [])[start_ind:end_ind],
            'pressure': dataset.get('pressure', [])[start_ind:end_ind],
            'pressure_bc': dataset.get('pressure_bc', [])[start_ind:end_ind],
            'samp_rate': dataset.get('samp_rate', None),
            'nl': dataset.get('nl', None)
        }
        result = func(window_dataset, **kwargs)
        results.append((start_ind, end_ind, result))
        mid_points.append((start_ind + end_ind) // 2)
        
        start_ind += step_size
        end_ind += step_size

    # Handle the last window if it doesn't fit perfectly
    if data_len - start_ind >= window_length // 2:
        end_ind = data_len

        window_dataset = {
            'time': dataset.get('time', [])[start_ind:end_ind],
            'pressure': dataset.get('pressure', [])[start_ind:end_ind],
            'pressure_bc': dataset.get('pressure_bc', [])[start_ind:end_ind],
            'samp_rate': dataset.get('samp_rate', None),
            'nl': dataset.get('nl', None)
        }
        result = func(window_dataset, **kwargs)
        results.append((start_ind, end_ind, result))
        mid_points.append((start_ind + end_ind) // 2)

    elif data_len - start_ind < window_length // 2 and data_len - start_ind > 0:
        start_ind -= step_size  # Move back to ensure we have enough data for the last window
        end_ind = data_len

        window_dataset = {
            'time': dataset.get('time', [])[start_ind:end_ind],
            'pressure': dataset.get('pressure', [])[start_ind:end_ind],
            'pressure_bc': dataset.get('pressure_bc', [])[start_ind:end_ind],
            'samp_rate': dataset.get('samp_rate', None),
            'nl': dataset.get('nl', None)
        }
        result = func(window_dataset, **kwargs)
        results[-1] = (start_ind, end_ind, result)  # Replace the last result with the final window result
        mid_points[-1] = (start_ind + end_ind) // 2  # Update the last mid-point
    # Combine results from all windows
    combined_result = np.zeros(data_len)

    for mid_point, (start_ind, end_ind, result) in zip(mid_points, results):
        combined_result[mid_point] = result
    
    # Interpolate between mid-points
    for i in range(len(mid_points) - 1):
        start_mid = mid_points[i]
        end_mid = mid_points[i + 1]
        start_val = combined_result[start_mid]
        end_val = combined_result[end_mid]
        
        # Linear interpolation
        indices = np.arange(start_mid, end_mid + 1)
        interpolated = start_val + (end_val - start_val) * (indices - start_mid) / (end_mid - start_mid)
        combined_result[start_mid:end_mid + 1] = interpolated
    
    # Extrapolate at the beginning and end
    if mid_points[0] > 0:
        first_slope = (combined_result[mid_points[1]] - combined_result[mid_points[0]]) / (mid_points[1] - mid_points[0])
        idxs = np.arange(mid_points[0])
        combined_result[:mid_points[0]] = combined_result[mid_points[0]] - first_slope * (mid_points[0] - idxs)

    if mid_points[-1] < data_len - 1:
        last_slope = (combined_result[mid_points[-1]] - combined_result[mid_points[-2]]) / (mid_points[-1] - mid_points[-2])
        idxs = np.arange(mid_points[-1] + 1, data_len)
        combined_result[mid_points[-1] + 1:] = combined_result[mid_points[-1]] + last_slope * (idxs - mid_points[-1])
    
    return combined_result

def find_baseline(dataset, bl_parameters):
    """
    Function to find the baseline curve for the pressure data, which can be used for baseline correction.

    Parameters:
    -----------
    bl_parameters : dict
        Parameters required for baseline finding.

    Returns:
    --------
    array-like
        The estimated baseline curve.
    """

    n = bl_parameters['n']
    pressure = dataset.get('pressure', [])
    pressure_len = len(pressure)
    baselines_vals = []

    hist_range = BL_HIST_RES * AMP_DISCRATISATION / 2
    counts, bins = np.histogram(pressure, bins=BL_HIST_RES, range=(-hist_range, hist_range))
    max_index = counts.argmax()
    baseline_val = bins[max_index]

    return baseline_val

    # for j in range(n):
    #     window_num = 2**j  # Exponential number of windows
    #     # Calculate window size to fit all windows with 50% overlap spanning the entire data
    #     # With 50% overlap: total_span = window_size + (window_size/2) * (num_windows - 1)
    #     # Solving for window_size: window_size * (num_windows + 1) / 2 = pressure_len
    #     window_size = int(pressure_len * 2 / (window_num + 1))
    #     step_size = window_size // 2  # 50% overlap means step is half the window size
        
    #     baselines = []
        
    #     for i in range(window_num):
    #         start_idx = i * step_size
    #         end_idx = start_idx + window_size
            
    #         # Ensure we don't exceed the data bounds
    #         if start_idx >= pressure_len:
    #             break
    #         if end_idx > pressure_len:
    #             end_idx = pressure_len
            
    #         # Extract window and calculate baseline
    #         window_data = pressure[start_idx:end_idx]
            
    #         if len(window_data) > 0:
    #             # Create histogram for this window with 1000 bins
    #             counts, bins = np.histogram(window_data, bins=100)
    #             max_index = counts.argmax()
    #             counts, bins = np.histogram(window_data, bins=10000, range=(bins[max_index],
    #                                                                         bins[max_index + 1]))
    #             bins = np.mean([bins[:-1], bins[1:]], axis=0)
                
    #             # Find the maximum of the histogram
    #             max_index = counts.argmax()
    #             baseline = bins[max_index]
    #             baselines.append(baseline)
        
    #     baselines_vals.append(baselines)

    # all_baselines = []
    # baselines_weights = []
    # total_weight = 0

    # for baseline_idx, baseline_list in enumerate(baselines_vals):
    #     # Number of windows for this baseline set
    #     window_num = len(baseline_list)

    #     total_weight += baseline_idx + 1  # Weight is proportional to the number of windows (more windows = more weight)
    #     baselines_weights.insert(0, baseline_idx + 1)  # Insert at beginning to have weights in order of increasing windows
    #     # fix
        
    #     # Calculate window positions - find middle indices of each window
    #     window_size = int(pressure_len * 2 / (window_num + 1))
    #     step_size = window_size // 2
        
    #     window_middle_indices = []
    #     for i in range(window_num):
    #         start_idx = i * step_size
    #         end_idx = start_idx + window_size
    #         if end_idx > pressure_len:
    #             end_idx = pressure_len
    #         middle_idx = (start_idx + end_idx) // 2
    #         window_middle_indices.append(middle_idx)
        
    #     # Create full baseline curve with interpolation
    #     baseline_curve = np.zeros(pressure_len)
        
    #     # Calculate first segment slope (for extrapolation at beginning)
    #     if window_num > 1:
    #         first_slope = (baseline_list[1] - baseline_list[0]) / (window_middle_indices[1] - window_middle_indices[0])
    #     else:
    #         first_slope = 0
        
    #     # Calculate last segment slope (for extrapolation at end)
    #     if window_num > 1:
    #         last_slope = (baseline_list[-1] - baseline_list[-2]) / (window_middle_indices[-1] - window_middle_indices[-2])
    #     else:
    #         last_slope = 0
        
    #     # Fill the beginning with extrapolation using first slope
    #     if window_middle_indices[0] > 0:
    #         idxs = np.arange(window_middle_indices[0])
    #         baseline_curve[:window_middle_indices[0]] = baseline_list[0] - first_slope * (window_middle_indices[0] - idxs)
        
    #     # Interpolate between window midpoints
    #     for i in range(len(window_middle_indices) - 1):
    #         start_idx = window_middle_indices[i]
    #         end_idx = window_middle_indices[i + 1]
    #         start_val = baseline_list[i]
    #         end_val = baseline_list[i + 1]
            
    #         # Linear interpolation
    #         indices = np.arange(start_idx, end_idx + 1)
    #         interpolated = start_val + (end_val - start_val) * (indices - start_idx) / (end_idx - start_idx)
    #         baseline_curve[start_idx:end_idx + 1] = interpolated
        
    #     # Fill the end with extrapolation using last slope
    #     if window_middle_indices[-1] + 1 < pressure_len:
    #         idxs = np.arange(window_middle_indices[-1] + 1, pressure_len)
    #         baseline_curve[window_middle_indices[-1] + 1:] = baseline_list[-1] + last_slope * (idxs - window_middle_indices[-1])
        
    #     all_baselines.append(baseline_curve)

    # # Calculate weighted average baseline
    # baseline=np.zeros(pressure_len)
    # for i in range(len(all_baselines)):
    #     baseline += all_baselines[i] * (baselines_weights[i] / total_weight)

    # return baseline

def find_peak_th(dataset, peak_th_parameters): 
    """
    Function to find the amplitude thresholds (both for inhale and exhale) for the pressure data,
    which can be used for breath analysis.

    Parameters:
    -----------
    peak_th_parameters : dict
        Parameters required for peak threshold finding.

    Returns:
    --------
    tuple of array-like
        The estimated inhalation and exhalation amplitude thresholds (inh_amp_th, exh_amp_th).
    """
    
    return dataset['nl']

def find_trough_th(dataset, trough_th_parameters): 
    """
    Function to find the amplitude thresholds (both for inhale and exhale) for the pressure data,
    which can be used for breath analysis.

    Parameters:
    -----------
    trough_th_parameters : dict
        Parameters required for trough threshold finding.

    Returns:
    --------
    tuple of array-like
        The estimated inhalation and exhalation amplitude thresholds (inh_amp_th, exh_amp_th).
    """
    
    return -dataset['nl']

def find_time_th(dataset, time_th_parameters):
    """
    Function to find the time threshold (TTH) for the pressure data, which can be used for breath analysis.

    Parameters:
    -----------
    tth_parameters : dict
        Parameters required for TTH finding.

    Returns:
    --------
    array-like
        The estimated TTH values.
    """
    
    pressure = dataset['pressure_bc']
    samp_rate = dataset['samp_rate']

    # Find the dominant inhale frequency using FFT
    n = len(pressure)
    fft_vals = np.abs(np.fft.rfft(pressure))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / samp_rate)

    # Exclude DC component (index 0)
    fft_vals[fft_freqs < DC_EXCLUSION_FREQ] = 0
    dominant_freq = fft_freqs[np.argmax(fft_vals)]
    period_samples = 1 / dominant_freq
    time_th = 0.5 * period_samples / TIME_THRESHOLD_DIVISOR # Half the period

    return time_th




