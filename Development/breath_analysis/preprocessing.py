import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import (INITIAL_UPSAMPLE, SECUNDO_CONST, DC_EXCLUSION_FREQ, TIME_THRESHOLD_DIVISOR,
                       WINDOW_NUM, BL_HIST_RES, AMP_DISCRATISATION, BASELINE_WINDOW, INH_AMP_WINDOW, EXH_AMP_WINDOW)

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
        dataset['amp_discratisation'], dataset['nl'] = nl
        dataset['pressure'], dataset['time'] = upsample(dataset['time_raw'], dataset['pressure_raw'], INITIAL_UPSAMPLE)
        dataset['samp_rate'] = INITIAL_UPSAMPLE
        dataset['baseline'] = WMD(dataset, {'window_length':BASELINE_WINDOW, 'overlap':50},
                                  find_baseline, bl_parameters={'n':WINDOW_NUM })
        # dataset['baseline'] = find_baseline(dataset, {'n': 5})
        dataset['pressure_bc'] = dataset['pressure'] - dataset['baseline']
        dataset['inh_amp_th'] = WMD(dataset, {'window_length':INH_AMP_WINDOW, 'overlap':50}, find_peak_th, peak_th_parameters={})
        dataset['exh_amp_th'] = WMD(dataset, {'window_length':EXH_AMP_WINDOW, 'overlap':50}, find_trough_th, trough_th_parameters={})
        # dataset['inh_amp_th'] = nl
        # dataset['exh_amp_th'] = -nl
        # dataset['time_th'] = WMD(dataset, {'window_length':1000, 'overlap':50},
                                 # find_time_th, time_th_parameters={})
        dataset['time_th'] = find_time_th(dataset, {})
    
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
    amp_discratisation = np.min(noise_range)
    nl = amp_discratisation * SECUNDO_CONST

    return amp_discratisation, nl
    
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
            'amp_discratisation': dataset.get('amp_discratisation', None),
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
            'amp_discratisation': dataset.get('amp_discratisation', None),
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
            'amp_discratisation': dataset.get('amp_discratisation', None),
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

    amp_discratisation = dataset['amp_discratisation']
    pressure = dataset.get('pressure', [])

    hist_range = BL_HIST_RES * amp_discratisation / 2
    counts, bins = np.histogram(pressure, bins=BL_HIST_RES, range=(-hist_range, hist_range))
    max_index = counts.argmax()
    baseline_val = bins[max_index]

    return baseline_val

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




