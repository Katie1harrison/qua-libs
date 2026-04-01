from typing import Optional

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    """Number of averages per (amplitude, tau) point."""
    mode_name: str = "alice"
    """Which cavity mode to calibrate: 'alice' or 'bob'."""
    amp_min: float = 0.0
    """Minimum displacement amplitude_scale (inclusive)."""
    amp_max: float = 1.0
    """Maximum displacement amplitude_scale (inclusive)."""
    amp_points: int = 11
    """Number of displacement amplitude points (linearly spaced)."""
    tau_min_ns: int = 100
    """Minimum Ramsey delay [ns]. Must be a multiple of 4."""
    tau_max_ns: int = 5000
    """Maximum Ramsey delay [ns]. Must be a multiple of 4."""
    tau_points: int = 51
    """Number of Ramsey delay points."""
    chi_hz: Optional[float] = None
    """Dispersive shift (peak spacing) in Hz. If None, read from cavity_transmon_pairs or
    cavity_mode.chi in the machine state. chi_hz = spacing between adjacent photon-number
    peaks = 2χ/(2π), where H = -χ a†a σz."""
    use_state_discrimination: bool = True
    """True → measure qubit state. False → measure raw I/Q."""
    displacement_pulse_duration_ns: Optional[int] = None
    """Override displacement pulse duration [ns]. None uses the operation default length."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
