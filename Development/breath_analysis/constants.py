

INITIAL_UPSAMPLE = 100
# --- Filtering ---
LOW_PASS_CUTOFF = 2         # Hz, cutoff frequency for low-pass Butterworth filter

# --- Baseline estimation ---
WINDOW_NUM = 1              # Number of window scales for baseline estimation

# --- Resampling ---
RESAMPLED_FREQ = 1000        # Hz, target frequency after upsampling

# --- Amplitude threshold ---
AMPLITUDE_THRESHOLD_DIVISOR = 3   # Median of positive values is divided by this

# --- SE threshold (histogram method) ---
SE_THRESHOLD_HIST_BINS  = 2000    # Number of bins for gap histogram
SE_THRESHOLD_HIST_RANGE = (2, 2000)  # Range (samples) for gap histogram
SE_THRESHOLD_HIST_DIVISOR = 3     # Most-common gap divided by this to get SE threshold

# --- FFT / DC exclusion ---
DC_EXCLUSION_FREQ = 0.04    # Hz, FFT components below this are zeroed (DC removal)
SE_THRESHOLD_FFT_DIVISOR = 3 # Dominant period divided by this to get SE threshold

# --- Outlier removal: duration ---
DURATION_LOW_THRESHOLD  = 0.15   # seconds, minimum valid breath duration
DURATION_HIGH_THRESHOLD = 10.0   # seconds, maximum valid breath duration

# --- Outlier removal: peak amplitude ---
PEAK_ZSCORE_THRESHOLD   = 2      # Z-score threshold for peak outlier detection
PEAK_BOUNDARY_ZSCORE    = 18     # Sentinel z-score forced at array edges (code debt)

# --- Event list construction ---
EVENT_SEARCH_WINDOW     = 30     # seconds, look-ahead/look-behind when finding event boundaries
MAX_POST_INHALE_PAUSE   = 2.0    # seconds, max inhale-exhale pause to count as one breath

MANUAL_BASELINE = 0
