import numpy as np

def zelano_parameters(dataset):
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
    pressure = dataset['pressure_upsampled']
    event_list = dataset['event_list']
    breath_list = dataset['breath_list']
    if len(breath_list) == 0:
        dataset['zelano_parameters'] = {
            'breathing_rate': 0,
            'inter_breath_interval': 0,
            'inhale_exhale_volumes': [],
            'tidal_volume': 0,
            'minute_ventilation': 0,
            'duty_cycle': 0,
            'cv_duty_cycle': 0,
            'cv_breathing_rate': 0,
            'cv_breath_volumes': 0
         }
        return dataset
    freq = dataset['upsampled_samp_rate']
    
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
        inhale_exhale_volumes.append(np.trapezoid(pressure[breath['start']:breath['end']], dx=1/freq))
    
    avg_inhale_volume = np.mean([np.trapezoid(pressure[event['start']:event['end']],
                                        dx=1/freq) for event in inhale_list])
    avg_exhale_volume = np.mean([np.trapezoid(pressure[event['start']:event['end']],
                                        dx=1/freq) for event in exhale_list])
    tidal_volume = avg_inhale_volume + avg_exhale_volume

    minute_ventilation = (breathing_rate * tidal_volume) * 60

    avg_inhale_duration = np.mean([event['duration'] for event in inhale_list])
    duty_cycle = avg_inhale_duration / inter_breath_interval

    cv_duty_cycle = np.std([event['duration'] for event in inhale_list]) / avg_inhale_duration

    cv_breathing_rate = np.std(np.diff([event['start'] for event in inhale_list]) / freq) / inter_breath_interval

    cv_breath_volumes = np.std(inhale_exhale_volumes) / np.mean(inhale_exhale_volumes)

    dataset['zelano_parameters'] = {
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
    return dataset