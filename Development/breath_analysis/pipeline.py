from preprocessing import read_data, preprocess_file
from extrimum_detection import find_extrimum_points

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

    data = read_data(file_path)
    data = preprocess_file(data)
    data = find_extremum_points(data)
    
    return data

if __name__ == "__main__":
    file_path = "/home/aviv/Work/Anat's Lab/Breath Analysis/data/failed_1.csv"
    results = analyze_file(file_path)