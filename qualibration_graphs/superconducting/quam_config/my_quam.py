from typing import Dict, Any, List, Optional
from quam.core import quam_dataclass, QuamBase
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam


@quam_dataclass
class TemporaryCalibrationData(QuamBase):
    """Temporary calibration data for a single qubit."""
    # abs(IQ) amplitudes in Volts for this qubit
    resonator_amplitudes: Dict[str, float] = None
    # Store arbitrary calibration parameters and metadata
    parameters: Dict[str, Any] = None
    # Adaptive frequency span for spectroscopy (in MHz)
    adaptive_frequency_span_mhz: Optional[float] = None
    # Adaptive power shift for spectroscopy (in dBm) - cumulative adjustment
    adaptive_power_shift_dbm: Optional[float] = None
    # Adaptive number of shots for weak peaks (legacy, no longer set)
    adaptive_num_shots: Optional[int] = None
    # Adaptive frequency step for zoom-in on weak peak (MHz) - set to 0.1 MHz after first weak peak
    adaptive_frequency_step_mhz: Optional[float] = None
    # Blacklisted qubit RF frequencies (Hz) that produced no Rabi oscillations
    blacklisted_qubit_frequencies: Optional[List[float]] = None
    # Blacklisted resonator RF frequencies (Hz) that produced no real dip
    blacklisted_resonator_frequencies: Optional[List[float]] = None
    # Initial (user-set) resonator frequencies saved before broad spectroscopy
    # overwrites them; restored if high-power spectroscopy fails
    initial_resonator_f01: Optional[float] = None
    initial_resonator_RF_frequency: Optional[float] = None
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
    resonator_amplitudes: Dict[str, Dict[str, float]] = None
