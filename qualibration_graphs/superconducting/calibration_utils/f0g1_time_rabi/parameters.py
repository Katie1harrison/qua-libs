from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    mode_name: str = "alice"
    """Which cavity mode to probe: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""

    min_duration_ns: int = 16
    """Minimum f0g1 pulse duration in ns. Must be a multiple of 4 (one clock cycle)."""
    max_duration_ns: int = 2000
    """Maximum f0g1 pulse duration in ns (exclusive). Must be a multiple of 4."""
    duration_step_ns: int = 4
    """Step size for the duration sweep in ns. Must be a multiple of 4."""

    operation: str = "f0g1_pi"
    """Operation to play on the f0g1 channel for the time Rabi sweep."""

    use_state_discrimination: bool = True
    """True -> measure qubit state (recommended). False -> measure raw I/Q."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
