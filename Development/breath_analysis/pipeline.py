import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import get_raw_data, low_pass_filter, get_baselines, get_weighted_average_baseline, upsample_data
from extrimum_detection import find_amp_threshold, find_se_threshold, find_se_points, find_peaks_durations, remove_outliers
from breath_parameters import pressure_switch, create_event_list, zelano_parameters

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

    if test_flag:
        results = get_raw_data(file_path)
        result = results[0]
        result = low_pass_filter(result)
        result = get_baselines(result)
        result = get_weighted_average_baseline(result)
        result = continue_analysis(result)
        return result
    else:
        results = analyze_file_to_baseline(file_path)
        for i, result in enumerate(results):
            results[i] = continue_analysis(result)
        # export_results(results)
        return results

def analyze_file_to_baseline(file_path):
    """
    Run the pipeline up to and including baseline calculation only.
    Returns a list of result dicts (one per pressure column), each containing
    'time', 'pressure_filtered', and 'baseline' — ready for display and
    optional user editing before calling continue_analysis().
    """
    results = get_raw_data(file_path)
    for i, result in enumerate(results):
        result = low_pass_filter(result)
        result = get_baselines(result)
        result = get_weighted_average_baseline(result)
        results[i] = result
    return results

def continue_analysis(result):
    """
    Run the post-baseline steps on a result that already has 'baseline' set.
    Recomputes pressure_bc from the current baseline (allowing for user edits),
    then runs upsample_data and analyze_breath.
    """
    result['pressure_bc'] = result['pressure_filtered'] - result['baseline']
    result = upsample_data(result)
    result = analyze_breath(result)
    return result

def analyze_breath(result):
    """
    Placeholder for any additional breath analysis steps that may be needed before thresholding.
    Currently does nothing but
    """
    inhale_parameters = {}
    
    inhale_parameters = find_amp_threshold(result, inhale_parameters, type_flag='inhale')
    inhale_parameters = find_se_threshold(result, inhale_parameters, type_flag='inhale')
    inhale_parameters = find_se_points(result, inhale_parameters, type_flag='inhale')
    inhale_parameters = find_peaks_durations(result, inhale_parameters, type_flag='inhale')
    inhale_parameters = remove_outliers(result, inhale_parameters)

    exhale_parameters = {}

    exhale_parameters = find_amp_threshold(result, exhale_parameters, type_flag='exhale')
    exhale_parameters = find_se_threshold(result, exhale_parameters, type_flag='exhale')
    exhale_parameters = find_se_points(result, exhale_parameters, type_flag='exhale')
    exhale_parameters = find_peaks_durations(result, exhale_parameters, type_flag='exhale')
    exhale_parameters = remove_outliers(result, exhale_parameters)

    exhale_parameters['parameters']['troughs'] = np.column_stack((exhale_parameters['parameters']['peaks'][:, 0],
                                                                   -exhale_parameters['parameters']['peaks'][:, 1]))
    del exhale_parameters['parameters']['peaks']
    exhale_parameters['threshold_dict']['amplitude_threshold'] = -exhale_parameters['threshold_dict']['amplitude_threshold']

    result['inhale_parameters'] = inhale_parameters
    result['exhale_parameters'] = exhale_parameters
    
    result = pressure_switch(result)
    result = create_event_list(result)
    result = zelano_parameters(result)
    
    return result

# def export_results(results):
#     """
#     Creates a new CSV file and exports the results of the breath analysis to it.
#     The file is saved next to the source file with ' - analysis' appended to the name.
#     Results are placed side by side: type - p1, start - p1, end - p1, duration - p1,
#     type - p2, start - p2, ...

#     Parameters:
#     -----------
#     results : list
#         List of result dicts from the breath analysis pipeline. Each dict must contain
#         'filename' and 'event_list'.
#     """
#     source_name = os.path.splitext(results[0]['filename'])[0]
#     output_file = source_name + ' - analysis.csv'

#     frames = []
#     for ind, result in enumerate(results):
#         label = f'p{ind + 1}'
#         df = pd.DataFrame(result['event_list'])[['type', 'start', 'end', 'duration']]
#         df.columns = [f'{col} - {label}' for col in df.columns]
#         frames.append(df)

#     combined = pd.concat(frames, axis=1)
#     combined.to_csv(output_file, index=False)

if __name__ == "__main__":
    results = analyze_file("/home/aviv/Work/Anat's Lab/Breath Analysis/data/data_20221106_073425_AA013.csv", test_flag=True)
   