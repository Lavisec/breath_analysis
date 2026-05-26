AMP_DISCRATISATION = 0.0166 # The amplitude discretisation of the meassurement.

# ----- Initial Upsample Frequency -----
INITIAL_UPSAMPLE = 100 # Hz, the initial upsample frequency to which all signals will be resampled before further processing.

# ----- Secundo Const -----
SECUNDO_CONST = 18 

# --- Baseline estimation ---
WINDOW_NUM = 5              # Number of window scales for baseline estimation
BL_HIST_RES = 1000          # Number of bins for histogram in baseline estimation (higher = more precise but slower)

# --- FFT / DC exclusion ---
DC_EXCLUSION_FREQ = 0.04    # Hz, FFT components below this are zeroed (DC removal)
TIME_THRESHOLD_DIVISOR = 3 # Dominant period divided by this to get SE threshold



