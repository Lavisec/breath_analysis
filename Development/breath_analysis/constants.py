AMP_DISCRATISATION = 0.0166 # The amplitude discretisation of the meassurement.

# ----- Initial Upsample Frequency -----
INITIAL_UPSAMPLE = 100 # Hz, the initial upsample frequency to which all signals will be resampled before further processing.

# ----- Windowing -----
BASELINE_WINDOW = 3600        # seconds, the window size for baseline estimation (code debt: should be in samples, not seconds)
INH_AMP_WINDOW = 1000         # seconds, the window size for inhalation amplitude estimation
EXH_AMP_WINDOW = 1000         # seconds, the window size for exhalation amplitude estimation

# ----- Secundo Const -----
SECUNDO_CONST = 18 

# --- Baseline estimation ---
WINDOW_NUM = 5              # Number of window scales for baseline estimation
BL_HIST_RES = 1000          # Number of bins for histogram in baseline estimation (higher = more precise but slower)

# --- FFT / DC exclusion ---
DC_EXCLUSION_FREQ = 0.04    # Hz, FFT components below this are zeroed (DC removal)
TIME_THRESHOLD_DIVISOR = 3 # Dominant period divided by this to get SE threshold

# --- Extrimum detection ---
SECOND_UPSAMPLE = 1000      # Hz, the upsample frequency for extrimum finding

# --- Outlier removal ---
DURATION_LOW_THRESHOLD  = 0.15   # seconds, minimum valid breath duration
DURATION_HIGH_THRESHOLD = 10.0   # seconds, maximum valid breath duration
PEAK_ZSCORE_THRESHOLD   = 2      # Z-score threshold for peak outlier detection
PEAK_BOUNDARY_ZSCORE    = 18     # Sentinel z-score forced at array edges (code debt)



