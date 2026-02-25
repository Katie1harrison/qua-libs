from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    # ------------------------------------------------------------------
    # User-facing power sweep (dBm)
    # ------------------------------------------------------------------
    min_power_dbm: float = -60
    max_power_dbm: float = -10
    num_power_points: int = 10

    # ------------------------------------------------------------------
    # OPX amplitude constraints
    # ------------------------------------------------------------------
    max_amplitude_opx: float = 0.1
    min_amplitude_opx: float = 0.01

    # ------------------------------------------------------------------
    # Spectroscopy parameters
    # ------------------------------------------------------------------
    frequency_span_in_mhz: float = 50
    frequency_step_in_mhz: float = 0.25

    use_adaptive_span: bool = False
    """
    Enable adaptive calibration adjustments for subsequent runs.
    When enabled, the node will automatically:
    - No peak found: Increase frequency span (up to 800 MHz) AND power (+10 dBm)
    - Weak peak at high power: Increase power only (+10 dBm), keep frequency span
    - Over-saturation: Decrease power (-10 dBm)
      (over-saturation = all power points show excessive linewidth or high baseline)
    """

    # ------------------------------------------------------------------
    # Averaging
    # ------------------------------------------------------------------
    num_shots: int = 100

    # ------------------------------------------------------------------
    # XY operation
    # ------------------------------------------------------------------
    operation: str = "saturation"
    operation_len_in_ns: Optional[int] = None

    # ------------------------------------------------------------------
    # Qubit peak analysis parameters
    # ------------------------------------------------------------------
    linewidth_threshold_hz: float = 2e6
    """
    Linewidth (FWHM) threshold above which the spectroscopy
    is considered power-broadened.
    """

    power_buffer_db: float = 3.0
    """
    Safety margin below the critical power (in dB).
    """

    min_peak_fraction: float = 0.1
    """
    Minimum acceptable peak height as a fraction of the difference.
    """

    peak_persistence_lookahead: int = 0
    """
    Number of consecutive higher-power levels that must also show a peak at a
    similar frequency for the current peak to be considered real.
    A peak that is absent from all of the next ``peak_persistence_lookahead``
    power levels is discarded as an isolated noise artefact.
    Set to 0 to disable persistence filtering entirely.
    """

    peak_persistence_freq_tolerance_hz: float = 5e6
    """
    Frequency tolerance (Hz) used when matching peaks across adjacent power
    levels during persistence filtering.  Two peaks at different power values
    are considered to be the same qubit transition if their detuning positions
    differ by less than this amount.
    """



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
