import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from constants import (
    AMPLITUDE_THRESHOLD_DIVISOR,
    SE_THRESHOLD_HIST_BINS, SE_THRESHOLD_HIST_RANGE, SE_THRESHOLD_HIST_DIVISOR,
    DC_EXCLUSION_FREQ, SE_THRESHOLD_FFT_DIVISOR,
    DURATION_LOW_THRESHOLD, DURATION_HIGH_THRESHOLD,
    PEAK_ZSCORE_THRESHOLD, PEAK_BOUNDARY_ZSCORE,
)

def find_amp_threshold(result, parameters, type_flag):

    
    if type_flag == 'inhale':
        pressure_resampled = result['pressure_resampled']
        
    else:    
        pressure_resampled = -result['pressure_resampled']
    
    # Define a threshold for inhale detection
    positive_vals = pressure_resampled[pressure_resampled > 0]
    amplitude_threshold = np.median(positive_vals) / AMPLITUDE_THRESHOLD_DIVISOR
    parameters['threshold_dict'] = {'amplitude_threshold': amplitude_threshold}
    
    return parameters

def find_se_threshold(result, parameters, type_flag):
    
    if type_flag == 'inhale':
        pressure_resampled = result['pressure_resampled']
        pressure_bc = result['pressure_bc']
    else:    
        pressure_resampled = -result['pressure_resampled']
        pressure_bc = -result['pressure_bc']
    resampled_freq = result['resampled_freq']
    freq_bc = result['samp_rate']
    
    amplitude_threshold = parameters['threshold_dict']['amplitude_threshold']
    
    # Find points above the threshold
    target_points = np.where(pressure_resampled > amplitude_threshold)[0]

    # Find the most common gap between above-threshold points to set the split threshold
    target_diffs = np.diff(target_points)
    target_diffs_hist = np.histogram(target_diffs, bins=SE_THRESHOLD_HIST_BINS, range=SE_THRESHOLD_HIST_RANGE) # Code debt
    most_common_diff = (target_diffs_hist[1][np.argmax(target_diffs_hist[0])]) / resampled_freq
    se_threshold_hist = most_common_diff / SE_THRESHOLD_HIST_DIVISOR
    parameters['threshold_dict']['se_threshold_hist'] = se_threshold_hist

    # Find the dominant inhale frequency using FFT
    n = len(pressure_bc)
    fft_vals = np.abs(np.fft.rfft(pressure_bc))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / freq_bc)

    # Exclude DC component (index 0)
    fft_vals[fft_freqs < DC_EXCLUSION_FREQ] = 0
    dominant_freq = fft_freqs[np.argmax(fft_vals)]
    target_period_samples = 1 / dominant_freq
    target_se_threshold_fft = 0.5 * target_period_samples / SE_THRESHOLD_FFT_DIVISOR # Half the period
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

    duration_low_threshold = DURATION_LOW_THRESHOLD
    duration_high_threshold = DURATION_HIGH_THRESHOLD

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
    
    first_z[0] = PEAK_BOUNDARY_ZSCORE # Code debt here
    last_z[-1] = PEAK_BOUNDARY_ZSCORE
    
    first_pos = np.where(first_z > PEAK_ZSCORE_THRESHOLD)[0][-1]
    last_pos = mid + np.where(last_z > PEAK_ZSCORE_THRESHOLD)[0][0]

    mask = np.zeros(len(sort_order), dtype=bool)
    mask[first_pos:last_pos + 1] = True
    delete_points_peaks = sort_order[~mask]
    parameters['outliers']['peak_outliers'] = delete_points_peaks

    parameters['parameters']['duration'] = np.delete(parameters['parameters']['duration'], delete_points_peaks, axis=0)
    parameters['parameters']['volume'] = np.delete(parameters['parameters']['volume'], delete_points_peaks, axis=0)
    parameters['parameters']['peaks'] = np.delete(parameters['parameters']['peaks'], delete_points_peaks, axis=0)
    parameters['se_points'] = np.delete(parameters['se_points'], delete_points_peaks, axis=0)

    return parameters
    