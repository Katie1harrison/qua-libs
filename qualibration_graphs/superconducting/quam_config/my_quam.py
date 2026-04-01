from typing import Dict, Any, List, Optional
from quam.core import quam_dataclass, QuamBase
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam


@quam_dataclass
class TemporaryCalibrationData(QuamBase):
    """Temporary calibration data for a single qubit."""
    # Store arbitrary calibration parameters and metadata
    parameters: Dict[str, Any] = None
    # Adaptive frequency span for spectroscopy (in MHz)
    adaptive_frequency_span_mhz: Optional[float] = None
    # Adaptive power shift for spectroscopy (in dBm) - cumulative adjustment
    adaptive_power_shift_dbm: Optional[float] = None
    # Adaptive number of shots (legacy, no longer set)
    adaptive_num_shots: Optional[int] = None
    # Blacklisted (qubit RF frequency Hz, drive power dBm) pairs that produced no Rabi oscillations.
    # Each entry is [freq_hz, power_dbm]; masking applies within ±5 MHz and ±3 dBm.
    blacklisted_qubit_points: Optional[List[List[float]]] = None
    # Blacklisted resonator RF frequencies (Hz) that produced no real dip
    blacklisted_resonator_frequencies: Optional[List[float]] = None
    # Initial (user-set) resonator frequencies saved before broad spectroscopy
    # overwrites them; restored if high-power spectroscopy fails
    initial_resonator_f01: Optional[float] = None
    initial_resonator_RF_frequency: Optional[float] = None
    # Initial x180 amplitude and qubit frequency saved at the start of x180 fine calibration;
    # restored if a Rabi or Ramsey fit fails mid-loop
    initial_x180_amplitude: Optional[float] = None
    initial_qubit_f01: Optional[float] = None
    # Initial LO/RF frequency (Hz) saved alongside initial_qubit_f01;
    # restored together with f_01 if a Rabi or Ramsey fit fails mid-loop
    initial_rf_frequency: Optional[float] = None
    # Selected spectroscopy drive power (dBm) and Octave gain (dB) saved by
    # qubit_spectroscopy_vs_power on success; used by downstream nodes to know the
    # power setting at which the qubit was found
    selected_power_dbm: Optional[float] = None
    selected_octave_gain_db: Optional[float] = None
    # Adaptive x180 pulse duration (ns) used when amplitude + Octave gain are both at maximum
    # and TOO_FEW_PERIODS is detected.  None means no duration adaptation is active.
    adaptive_x180_length_ns: Optional[float] = None
    # Original x180 pulse length (ns) saved at the start of duration adaptation;
    # restored if a Rabi fit fails while duration adaptation is active
    initial_x180_length_ns: Optional[float] = None
    # Optional timestamp or metadata fields
    last_updated: Optional[str] = None
    notes: Optional[str] = None


# Define the QUAM class that will be used in all calibration nodes
# Should inherit from either FixedFrequencyQuam or FluxTunableQuam
@quam_dataclass
class Quam(FixedFrequencyQuam):
    # Temporary calibration data per qubit
    # Note: Use Dict for QUAM JSON serialization compatibility
    temp_calibration: Dict[str, TemporaryCalibrationData] = None
