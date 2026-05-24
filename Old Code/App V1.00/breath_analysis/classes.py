import numpy as np


class BreathRecording:
    """
    Represents all data and results for a single pressure-sensor channel from one recording session.

    A BreathRecording is created from raw CSV data and is progressively enriched by the
    processing pipeline (filtering → baseline correction → upsampling → event detection →
    parameter extraction). It acts as the central data container that replaces the
    plain `result` dict used in the functional pipeline.

    Attributes
    ----------
    # --- Raw / provenance ---
    filename : str
        Name of the source CSV file.
    channel : str
        Pressure-column name this recording was derived from (e.g. 'pressure1').

    # --- Raw signal ---
    time : np.ndarray
        Uniformly-spaced time axis (seconds) after initial interpolation.
    samp_rate : float
        Sampling rate of the raw signal (Hz).
    pressure_raw : np.ndarray
        Raw pressure values on the interpolated time axis.

    # --- Filtered signal ---
    pressure_filtered : np.ndarray | None
        Low-pass filtered pressure signal. Set by low_pass_filter().

    # --- Baseline correction ---
    baselines : list[list[float]] | None
        Per-window baseline values produced by get_baselines().
    baseline : np.ndarray | None
        Weighted-average baseline curve. Set by get_weighted_average_baseline().
    pressure_bc : np.ndarray | None
        Baseline-corrected pressure (pressure_filtered − baseline).

    # --- Resampled signal ---
    time_resampled : np.ndarray | None
        Uniformly-spaced time axis at resampled_freq resolution.
    resampled_freq : float | None
        Upsampled frequency (Hz). Set by upsample_data().
    pressure_resampled : np.ndarray | None
        Baseline-corrected signal resampled to resampled_freq.

    # --- Detection intermediates ---
    inhale_parameters : dict | None
        Threshold values, SE points, peaks, and per-inhale stats produced by the
        inhale branch of the detection pipeline.
    exhale_parameters : dict | None
        Same structure as inhale_parameters but for the exhale branch.

    # --- Events & breaths ---
    event_list : list[dict] | None
        Ordered list of event dicts (inhale / exhale / pause) with keys:
        'start', 'end', 'duration', 'type'.
    breath_list : list[dict] | None
        Subset of event_list grouped into complete breaths. Each entry has keys:
        'inhale', 'exhale', 'pause', 'start', 'end', 'duration'.

    # --- High-level parameters ---
    zelano_parameters : dict | None
        Breathing-rate statistics as defined in Zelano et al. 2016.
    """

    def __init__(self, filename: str, channel: str,
                 time: np.ndarray, samp_rate: float, pressure_raw: np.ndarray):
        # Provenance
        self.filename = filename
        self.channel = channel

        # Raw signal
        self.time = time
        self.samp_rate = samp_rate
        self.pressure_raw = pressure_raw

        # Filtered
        self.pressure_filtered: np.ndarray | None = None

        # Baseline
        self.baselines: list | None = None
        self.baseline: np.ndarray | None = None
        self.pressure_bc: np.ndarray | None = None

        # Resampled
        self.time_resampled: np.ndarray | None = None
        self.resampled_freq: float | None = None
        self.pressure_resampled: np.ndarray | None = None

        # Detection intermediates
        self.inhale_parameters: dict | None = None
        self.exhale_parameters: dict | None = None

        # Events
        self.event_list: list | None = None
        self.breath_list: list | None = None

        # High-level parameters
        self.zelano_parameters: dict | None = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        duration = f"{self.time[-1] - self.time[0]:.1f} s" if self.time is not None else "?"
        n_breaths = len(self.breath_list) if self.breath_list is not None else "?"
        return (f"BreathRecording(file={self.filename!r}, channel={self.channel!r}, "
                f"duration={duration}, breaths={n_breaths})")

    @property
    def duration(self) -> float:
        """Total recording duration in seconds (raw time axis)."""
        return float(self.time[-1] - self.time[0])

    @property
    def n_breaths(self) -> int:
        """Number of complete detected breaths, or 0 if not yet computed."""
        return len(self.breath_list) if self.breath_list is not None else 0


# ---------------------------------------------------------------------------


class BreathEvent:
    """
    Represents a single detected respiratory event within a BreathRecording.

    Events are the atomic units that make up `BreathRecording.event_list` and the
    sub-components of each entry in `BreathRecording.breath_list`. Possible types are
    'inhale', 'exhale', and 'pause (inhale - exhale)' / 'pause (exhale - inhale)'.

    Attributes
    ----------
    type : str
        Event type: 'inhale', 'exhale', or a pause descriptor string.
    start : int
        Index (into the resampled time axis) where the event begins.
    end : int
        Index (into the resampled time axis) where the event ends.
    duration : float
        Duration of the event in seconds.

    # --- Optional sub-structure (complete breaths only) ---
    inhale : BreathEvent | None
        The inhale component when this event represents a complete breath.
    exhale : BreathEvent | None
        The exhale component when this event represents a complete breath.
    pause : BreathEvent | None
        The post-inhale pause component when this event represents a complete breath.
    """

    def __init__(self, type: str, start: int, end: int, duration: float):
        self.type = type
        self.start = start
        self.end = end
        self.duration = duration

        # Sub-structure for complete breath events
        self.inhale: "BreathEvent | None" = None
        self.exhale: "BreathEvent | None" = None
        self.pause: "BreathEvent | None" = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"BreathEvent(type={self.type!r}, start={self.start}, "
                f"end={self.end}, duration={self.duration:.3f} s)")

    @property
    def is_inhale(self) -> bool:
        return self.type == 'inhale'

    @property
    def is_exhale(self) -> bool:
        return self.type == 'exhale'

    @property
    def is_pause(self) -> bool:
        return self.type.startswith('pause')

    @property
    def is_complete_breath(self) -> bool:
        """True when this event bundles an inhale, pause, and exhale together."""
        return self.inhale is not None and self.exhale is not None

    @classmethod
    def from_dict(cls, d: dict) -> "BreathEvent":
        """
        Construct a BreathEvent from one of the plain dicts currently produced by
        create_event_list(), making migration from the functional pipeline straightforward.

        Parameters
        ----------
        d : dict
            A dict with at minimum the keys 'type', 'start', 'end', 'duration'.
            If it also has 'inhale', 'exhale', and 'pause' keys (complete-breath dicts),
            those sub-components are converted recursively.
        """
        event = cls(
            type=d['type'],
            start=d['start'],
            end=d['end'],
            duration=d['duration'],
        )
        if 'inhale' in d:
            event.inhale = cls.from_dict(d['inhale'])
        if 'exhale' in d:
            event.exhale = cls.from_dict(d['exhale'])
        if 'pause' in d:
            event.pause = cls.from_dict(d['pause'])
        return event

    def to_dict(self) -> dict:
        """Serialize back to the plain-dict format used by the functional pipeline."""
        d = {
            'type': self.type,
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
        }
        if self.inhale is not None:
            d['inhale'] = self.inhale.to_dict()
        if self.exhale is not None:
            d['exhale'] = self.exhale.to_dict()
        if self.pause is not None:
            d['pause'] = self.pause.to_dict()
        return d
