import numpy as np

from constants import (EVENT_SEARCH_WINDOW, MAX_POST_INHALE_PAUSE
                       ,DURATION_LOW_THRESHOLD, DURATION_HIGH_THRESHOLD)

def pressure_switch(result):
    result['pressure_resampled_lp'] = result['pressure_resampled']
    pressure_raw_bc = result['pressure_raw'] - result['baseline']
    result['pressure_resampled'] = np.interp(result['time_resampled'],
                                             result['time'], pressure_raw_bc)
    return result
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
            left_start = 0 if int(ev[0]) < EVENT_SEARCH_WINDOW*freq else int(ev[0]) - int(EVENT_SEARCH_WINDOW*freq)
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
            right_end = min(int(ev[0]) + int(EVENT_SEARCH_WINDOW*freq), len(pressure))
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
            left_start = 0 if int(ev[0]) < EVENT_SEARCH_WINDOW*freq else int(ev[0]) - int(EVENT_SEARCH_WINDOW*freq)
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
            right_end = min(int(ev[0]) + int(EVENT_SEARCH_WINDOW*freq), len(pressure))
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
            
            duration = (end - start) / freq
            if duration < DURATION_LOW_THRESHOLD or duration > DURATION_HIGH_THRESHOLD:
                continue
        
        event = {
            'start': start,
            'end': end,
            'type': type,
            'duration': (end - start) / freq,
            'extrimum': ev,
            'volume': np.trapezoid(pressure[start:end], dx=1/freq)
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

    breath_list = []
    for ind in range(len(event_list)):
        if event_list[ind]['type'] == 'pause (inhale - exhale)' and event_list[ind]['duration'] < MAX_POST_INHALE_PAUSE:
            breath_list.append({
                'inhale': event_list[ind - 1],
                'exhale': event_list[ind + 1],
                'pause': event_list[ind],
                'start': event_list[ind - 1]['start'],
                'end': event_list[ind + 1]['end'],
                'duration': (event_list[ind + 1]['end'] - event_list[ind - 1]['start']) / freq
            })
            
    
    result['event_list'] = event_list
    result['breath_list'] = breath_list
    return result

def zelano_parameters(result):
    """
    Calculate the parameters defined in Zelano et al. 2018 for the given result dictionary. This includes: the breathing rate, 
    the inter-breath interval, the inhale/exhale volumes, tidal volume, minute ventilation, duty cicle, coefficient of variation of duty cycle,
    coefficient of variation of breathing rate and coefficient of variation of breath volumes.

    These are the definition of the parameters acording to Zelano et al. 2018:

    - Breathing rate: 1/average time between inhale onsets
    - Inter-breath interval: Average time between inhale onsets
    - Inhale/exhale volumes: Sum of airflow between breath onset and offset
    - Tidal volume: Average inhale volume + average exhale volume
    - Minute ventilation: (Breathing rate × average tidal volume)/1 min
    - Duty cycle: Average inhale duration/average interbreath interval
    - Coefficient of variation of duty cycle: SD of inhale duration/average inhale duration
    - Coefficient of variation of breathing rate: SD of difference between inhale onsets/average difference between inhale onsets
    - Coefficient of variation of breath volumes: SD of breath volumes/average breath volume
    """
    event_list = result['event_list']
    breath_list = result['breath_list']
    freq = result['resampled_freq']
    
    inhale_list = []
    exhale_list = []
    pause_list = []

    for event in event_list:
        if event['type'] == 'inhale':
            inhale_list.append(event)
        elif event['type'] == 'exhale':
            exhale_list.append(event)
        else:
            pause_list.append(event)
    
    inter_breath_interval = np.mean([(inhale_list[i + 1]['start'] - inhale_list[i]['start']) / freq for i in range(len(inhale_list) - 1)])
    breathing_rate = 1 / inter_breath_interval

    inhale_exhale_volumes = []
    for breath in breath_list:
        inhale_exhale_volumes.append(np.trapezoid(result['pressure_resampled'][breath['start']:breath['end']], dx=1/freq))
    
    avg_inhale_volume = np.mean([np.trapezoid(result['pressure_resampled'][event['start']:event['end']],
                                           dx=1/result['resampled_freq']) for event in inhale_list])
    avg_exhale_volume = np.mean([np.trapezoid(result['pressure_resampled'][event['start']:event['end']],
                                           dx=1/result['resampled_freq']) for event in exhale_list])
    tidal_volume = avg_inhale_volume + avg_exhale_volume

    minute_ventilation = (breathing_rate * tidal_volume) * 60

    avg_inhale_duration = np.mean([event['duration'] for event in inhale_list])
    duty_cycle = avg_inhale_duration / inter_breath_interval

    cv_duty_cycle = np.std([event['duration'] for event in inhale_list]) / avg_inhale_duration

    cv_breathing_rate = np.std(np.diff([event['start'] for event in inhale_list]) / freq) / inter_breath_interval

    cv_breath_volumes = np.std(inhale_exhale_volumes) / np.mean(inhale_exhale_volumes)

    result['zelano_parameters'] = {
        'breathing_rate': breathing_rate,
        'inter_breath_interval': inter_breath_interval,
        'inhale_exhale_volumes': inhale_exhale_volumes,
        'tidal_volume': tidal_volume,
        'minute_ventilation': minute_ventilation,
        'duty_cycle': duty_cycle,
        'cv_duty_cycle': cv_duty_cycle,
        'cv_breathing_rate': cv_breathing_rate,
        'cv_breath_volumes': cv_breath_volumes
    }
    return result

# def recreate_baseline





















