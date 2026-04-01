from pydantic import field_validator

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters

_AMP_MAX = 2.0 - 2**-16  # QUA hardware limit


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages."""
    mode_name: str = "alice"
    """Which cavity mode to probe: 'alice' or 'bob'."""
    displacement_scale: float = 1.0
    """Amplitude scale for the cavity displacement pulse.
    Creates a coherent state |α⟩ with mean photon number n̄ = displacement_scale²
    (after node 32 calibration: scale=1 → 1 photon).
    Must be in (-2, +2) — the QUA hardware limit is ±(2 - 2^-16)."""
    active_reset: bool = True
    """If True, apply a reverse displacement D(-α) after measurement to
    return the cavity to vacuum immediately, replacing passive thermalization.
    Only valid when using displacement (not saturation) to populate the cavity."""
    right_offset_mhz: float = 2.0
    """Frequency offset above the qubit ge frequency where the sweep starts [MHz].
    A small positive margin above the n=0 peak (which sits at the qubit frequency)."""
    left_span_mhz: float = 6.0
    """Frequency span below the qubit ge frequency to sweep [MHz].
    Should be large enough to cover all visible photon-number peaks: left_span > max_peaks * chi."""
    frequency_step_in_mhz: float = 0.04
    """Frequency step [MHz]."""
    max_peaks: int = 8
    """Maximum number of photon-number peaks to try when fitting."""
    chi2_threshold: float = 2.0
    """Reduced chi² threshold for accepting a multi-Gaussian fit."""
    qubit_pulse: str = "selective_x180"
    """Qubit pulse operation to use for spectroscopy.
    Typical choices: 'selective_x180' (narrow-bandwidth, resolves photon-number peaks)
    or 'x180' (standard pi-pulse, faster but lower frequency resolution)."""
    use_state_discrimination: bool = True
    """True -> measure qubit state. False -> measure raw I/Q."""

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
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
