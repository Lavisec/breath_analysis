import numpy as np
from scipy import stats

from preprocessing import upsample
from constants import (SECOND_UPSAMPLE, DURATION_LOW_THRESHOLD, DURATION_HIGH_THRESHOLD,
                       PEAK_ZSCORE_THRESHOLD, PEAK_BOUNDARY_ZSCORE)

def find_extremum_points(data):
    """
    Detect extremum points (peaks and troughs) in the pressure signal.

    Parameters:
    -----------
    data : dict
        Dictionary containing 'time', 'pressure_filtered', 'inh_amp_th', and 'exh_amp_th'.

    Returns:
    --------
    data : dict
        Updated data dictionary with 'inhale_se_points' and 'exhale_se_points' added as lists of tuples
        containing the start and end indices of each detected inhale and exhale.
    """
    for dataset in data:
        dataset['pressure_upsampled'], dataset['time_upsampled'] = upsample(dataset['time'], 
                                                                                   dataset['pressure'], SECOND_UPSAMPLE)
        dataset['upsampled_samp_rate'] = SECOND_UPSAMPLE
        # dataset['peaks'], dataset['troughs'] = find_extrema_scipy(dataset)
        dataset['peaks'], dataset['troughs'] = find_extrema_se(dataset)
        dataset['event_list'] = create_event_list(dataset)
        dataset = remove_outliers(dataset)
        
    return data

def find_extrema_scipy(dataset):
    """
    Use scipy's find_peaks to detect peaks and troughs in the upsampled pressure signal.

    Parameters:
    -----------
    dataset : dict
        Dictionary containing 'pressure_upsampled' and 'time_upsampled'.

    Returns:
    --------
    peaks : list of tuples
        List of (start_index, end_index) for each detected peak (inhale).
    troughs : list of tuples
        List of (start_index, end_index) for each detected trough (exhale).
    """
    from scipy.signal import find_peaks
    
    # Fill in the logic to find peaks and troughs using scipy's find_peaks

def find_extrema_se(dataset):
    """
    Use the SE method to detect peaks and troughs in the upsampled pressure signal.

    Parameters:
    -----------
    dataset : dict
        Dictionary containing 'pressure_upsampled', 'time_upsampled', 'inh_amp_th', and 'exh_amp_th'.

    Returns:
    --------
    peaks : list of tuples
        List of (start_index, end_index) for each detected peak (inhale).
    troughs : list of tuples
        List of (start_index, end_index) for each detected trough (exhale).
    """
    peaks = []
    troughs = []

    for i, type in enumerate(['inhale', 'exhale']):
        pressure = (-1)**i * dataset['pressure_upsampled']
        time = dataset['time_upsampled']
        samp_rate = dataset['upsampled_samp_rate']
        amp_th = dataset['inh_amp_th'] if type == 'inhale' else -dataset['exh_amp_th']
        amp_th = np.interp(time, dataset['time'], amp_th)
        # time_th = np.interp(time, dataset['time'], dataset['time_th'])
        time_th = dataset['time_th'] * samp_rate

        target_points = np.where(pressure > amp_th)[0]
        target_diffs = np.diff(target_points)

        start_idx = target_points[np.where(target_diffs > time_th)[0] + 1] # Code debt
        start_idx = [target_points[0], *start_idx]
        end_idx = target_points[np.where(target_diffs > time_th)[0]]
        end_idx = [*end_idx, target_points[-1]]

        se_points = np.column_stack((start_idx, end_idx))

        # Dealing with cases where the start and end points are the same (e.g. when there is only one point above the threshold)
        del_points = np.where((se_points[:, 1] - se_points[:, 0]) == 0)[0]
        se_points = np.delete(se_points, del_points, axis=0)

        extrima_arr = np.zeros((len(se_points), 2))
        for j in range(len(se_points)):
            start = se_points[j, 0]
            end = se_points[j, 1]

            ind = np.argmax(pressure[start:end]) + start
            val = (-1)**i * pressure[ind]
            extrima_arr[j] = np.array([ind, val])
        
        if type == 'inhale':
            peaks = extrima_arr
        else:
            troughs = extrima_arr

    return peaks, troughs

def create_event_list(dataset):
    """
    Create a list of events (inhales, exhales and pauses) based on the detected peaks and troughs.

    Parameters:
    -----------
    dataset : dict
        Dictionary containing 'peaks' and 'troughs'.

    Returns:
    --------
    event_list : list of tuples
        List of (event_type, start_index, end_index) for each detected event, where event_type is 'inhale', 'exhale' or 'pause'.
    """
    peaks = dataset['peaks']
    troughs = dataset['troughs']
    pressure = dataset['pressure_upsampled']
    freq = dataset['upsampled_samp_rate']

    extrema_list = []
    event_list = []

    peaks_troughs = np.concatenate((peaks, troughs), axis=0)
    sorted_peaks_troughs = peaks_troughs[np.argsort(peaks_troughs[:, 0])]

    pressure_neg_idx = np.flatnonzero(pressure < 0)
    pressure_pos_idx = np.flatnonzero(pressure > 0)

    for ev in sorted_peaks_troughs:
        if ev[1] > 0:
            type = 'inhale'
            peak_ind = int(ev[0])

            left_first_neg = np.searchsorted(pressure_neg_idx, peak_ind) - 1
            left_last_pos = pressure_neg_idx[left_first_neg] + 1 if left_first_neg >= 0 else 0
            start = left_last_pos

            right_first_neg = np.searchsorted(pressure_neg_idx, peak_ind)
            right_last_pos = pressure_neg_idx[right_first_neg] - 1 if right_first_neg < len(pressure_neg_idx) else len(pressure) - 1
            end = right_last_pos
        else:
            type = 'exhale'
            trough_ind = int(ev[0])

            left_first_pos = np.searchsorted(pressure_pos_idx, trough_ind) - 1
            left_last_neg = pressure_pos_idx[left_first_pos] + 1 if left_first_pos >= 0 else 0
            start = left_last_neg

            right_first_pos = np.searchsorted(pressure_pos_idx, trough_ind)
            right_last_neg = pressure_pos_idx[right_first_pos] - 1 if right_first_pos < len(pressure_pos_idx) else len(pressure) - 1
            end = right_last_neg
        
        event = {
            'start': start,
            'end': end,
            'type': type,
            'duration': (end - start) / freq,
            'extrimum': ev,
            'volume': np.trapezoid(pressure[start:end], dx=1/freq)
        }
        extrema_list.append(event)

    for ind in range(len(extrema_list) - 1):
        event_list.append(extrema_list[ind])
        event_list.append({
            'start': extrema_list[ind]['end'],
            'end': extrema_list[ind + 1]['start'],
            'duration': (extrema_list[ind + 1]['start'] - extrema_list[ind]['end']) / freq,
            'type': 'pause'
        })
    event_list.append(extrema_list[-1])

    return event_list

def remove_outliers(dataset):
    """
    Remove outliers from the detected peaks and troughs based on their amplitude.

    Parameters:
    -----------
    dataset : dict
        Dictionary containing 'peaks' and 'troughs'.

    Returns:
    --------
    dataset : dict
        Updated dataset dictionary with outliers removed from 'peaks' and 'troughs'.
    """
    event_list = dataset['event_list']
    peaks = dataset['peaks']
    troughs = dataset['troughs']

    delete_points = []

    for i, extrema in enumerate([peaks, troughs]):
        extrema_pos = (-1)**i * extrema
        peak_values = extrema_pos[:, 1]
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
        delete_points.append(extrema[sort_order[~mask], 0])
    
    delete_points = np.concatenate(delete_points)

    for event in event_list:
        type = event['type']
        duration = event['duration']

        if type == 'inhale' or type == 'exhale':
            extrimum_idx = event['extrimum'][0]

            if duration < DURATION_LOW_THRESHOLD or duration > DURATION_HIGH_THRESHOLD:
                event['type'] = 'pause'
                continue

            if extrimum_idx in delete_points:
                event['type'] = 'pause'
                continue

    dataset['event_list'] = event_list
    
    return dataset




























