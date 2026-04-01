from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages."""
    min_amp_factor: float = 0.01
    """Minimum amplitude factor relative to the f0g1 saturation pulse amplitude."""
    max_amp_factor: float = 1.99
    """Maximum amplitude factor relative to the f0g1 saturation pulse amplitude."""
    amp_factor_step: float = 0.01
    """Step size for the amplitude factor sweep."""
    mode_name: str = "alice"
    """Which cavity mode to calibrate: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""
    operation: str = "f0g1_pi"
    """Name of the f0g1 operation whose amplitude will be calibrated."""
    use_state_discrimination: bool = True
    """True → measure qubit state (recommended). False → measure raw I/Q."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
