import matplotlib.pyplot as plt
import numpy as np

from preprocessing import read_data, preprocess_file, find_time_th
from extrimum_detection import find_extremum_points
from breath_analysis import zelano_parameters


def analyze_file_to_baseline(file_path):
    """
    Run the pipeline up to baseline calculation for a single CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to analyze.

    Returns
    -------
    list[dict]
        One result dict per pressure column, containing preprocessing outputs
        including baseline and baseline-corrected signal.
    """
    data = read_data(file_path)
    data = preprocess_file(data)
    return data


def continue_analysis(dataset):
    """
    Continue analysis from a preprocessed single-pressure result.

    This is intended for the GUI flow where baseline can be edited before
    finishing the analysis. We recompute dependent fields and then run
    extremum/event detection.

    Parameters
    ----------
    dataset : dict
        A single result dict produced by analyze_file_to_baseline.

    Returns
    -------
    dict
        The updated result dict with extremum and event outputs.
    """
    dataset['pressure_bc'] = dataset['pressure'] - dataset['baseline']
    dataset['time_th'] = find_time_th(dataset, {})
    dataset = find_extremum_points(dataset)
    dataset = zelano_parameters(dataset)
    return dataset


def analyze_file(file_path):
    """
    Run the full analysis pipeline on a single CSV file and return the results.
    This is the main entry point for the analysis.

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
        pressure_bc        : dict mapping column name -> baseline-corrected pressure array
        breath_metrics      : dict mapping column name -> dict of breath metrics (e.g. tidal volume, respiratory rate)
    """

    data = analyze_file_to_baseline(file_path)
    for i, dataset in enumerate(data):
        data[i] = continue_analysis(dataset)
    return data


def update_baseline(result, new_baseline):
    """
    Apply a new baseline to a result dict and re-run the post-baseline pipeline.

    Parameters
    ----------
    result : dict
        A single result dict produced by analyze_file_to_baseline.
    new_baseline : array-like
        The new baseline curve to apply.

    Returns
    -------
    dict
        The updated result dict with all downstream fields recomputed.
    """
    result['baseline'] = np.array(new_baseline, dtype=float)
    return continue_analysis(result)


def change_event_type(result, event_idx, new_type):
    """
    Change the type of a single event in the event list.

    Parameters
    ----------
    result : dict
        A result dict containing an 'event_list'.
    event_idx : int
        Index of the event to update.
    new_type : str
        New type string ('inhale', 'exhale', or 'pause').

    Returns
    -------
    dict
        The result dict with the event type updated (in-place, also returned).
    """
    result['event_list'][event_idx]['type'] = str(new_type).lower()
    return result


def update_event_bounds(result, event_idx, new_start=None, new_end=None):
    """
    Update the start and/or end sample index of an event, adjusting adjacent
    events to stay contiguous.

    When new_start is provided the end of the preceding event is set to
    new_start.  When new_end is provided the start of the following event is
    set to new_end.

    Constraints
    -----------
    * new_start is clamped to [start+1 of event N-1, end-1 of event N]
    * new_end   is clamped to [start+1 of event N, start-1 of event N+1]

    Parameters
    ----------
    result : dict
        A result dict containing an 'event_list' and 'time_upsampled'.
    event_idx : int
        Index of the event to update.
    new_start : int or None
        New start sample index, or None to leave unchanged.
    new_end : int or None
        New end sample index, or None to leave unchanged.

    Returns
    -------
    dict
        The result dict with the event list updated (in-place, also returned).
    """
    event_list = result['event_list']
    event = event_list[event_idx]

    if new_start is not None:
        # new_start must allow previous event to have at least 1 sample
        if event_idx > 0:
            min_start = int(event_list[event_idx - 1]['start']) + 1
        else:
            min_start = 0
        # new_start must allow current event to have at least 1 sample
        max_start = int(event['end']) - 1
        new_start = max(min_start, min(int(new_start), max_start))
        event['start'] = new_start
        if event_idx > 0:
            event_list[event_idx - 1]['end'] = new_start

    if new_end is not None:
        # new_end must allow current event to have at least 1 sample
        min_end = int(event['start']) + 1
        # new_end must allow next event to have at least 1 sample
        if event_idx < len(event_list) - 1:
            max_end = int(event_list[event_idx + 1]['end']) - 1
        else:
            max_end = len(result['time_upsampled']) - 1
        new_end = max(min_end, min(int(new_end), max_end))
        event['end'] = new_end
        if event_idx < len(event_list) - 1:
            event_list[event_idx + 1]['start'] = new_end

    return result

if __name__ == "__main__":
    file_path = "/home/aviv/Work/Anat's Lab/Breath Analysis/data/failed_1.csv"
    results = analyze_file(file_path)