from pydantic import field_validator
from typing import Literal

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
    IdleTimeNodeParameters,
)

_AMP_MAX = 2.0 - 2**-16  # QUA hardware limit


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of averages per time point."""
    mode_name: str = "alice"
    """Which cavity mode to probe: 'alice' or 'bob'."""
    displacement_scale: float = 1.0
    """Amplitude scale for the displacement pulse.
    After node 30/32 calibration: scale=1 → 1 photon.  Use scale>1 for a
    brighter initial coherent state (stronger signal at early times).
    Must be in (-2, +2) — the QUA hardware limit is ±(2 - 2^-16)."""
    delay_repeats: int = 1
    """Number of times to repeat the wait per point.  Extends the effective
    sweep range: total time = delay_repeats × t_per_rep × 4 ns.
    min/max_wait_time_in_ns define the *per-repeat* range, so the full sweep
    spans [min, delay_repeats × max] ns.  The x-axis always shows total time."""
    use_state_discrimination: bool = True
    """True → measure qubit state. False → measure raw I quadrature."""

    @field_validator("displacement_scale")
    @classmethod
    def _check_amp_scale(cls, v: float) -> float:
        if abs(v) > _AMP_MAX:
            raise ValueError(
                f"displacement_scale={v} exceeds the QUA hardware limit "
                f"±{_AMP_MAX:.6f} (2 - 2^-16). Use a value in (-2, +2)."
            )
        return v


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    # Override IdleTimeNodeParameters defaults to match cavity T1 timescales
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 5_000_000
    wait_time_num_points: int = 51
    log_or_linear_sweep: Literal["log", "linear"] = "log"
