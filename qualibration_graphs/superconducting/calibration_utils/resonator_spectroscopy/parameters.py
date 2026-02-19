from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters

from typing import Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 30.0
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    save_amplitudes: bool = False
    """Whether to save min/max resonator amplitudes to QUAM state. Default is False."""
    min_dip_contrast: float = 0.15
    """Minimum dip depth relative to baseline to consider a resonator dip real.
    The dip amplitude must be at least this fraction of the baseline.
    Set to 0 to disable. Default is 0.15 (15%)."""
    lo_leakage_exclusion_mhz: float = 10.0
    """Exclusion radius around the resonator LO upconversion frequency in MHz.
    Any detected dip within this distance of the LO frequency is rejected as an
    LO leakage artefact. Set to 0 to disable. Default is 10.0 MHz."""
    readout_power_dbm: Optional[float] = None
    """Readout power in dBm for the spectroscopy sweep.
    If None, the current QUAM state power is used unchanged.
    The QUAM state is reverted to its original value after the node finishes."""
    max_amp: float = 0.1
    """Maximum readout pulse amplitude (OPX units, 0–0.5).
    Only used when readout_power_dbm is set. Default is 0.1."""



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
