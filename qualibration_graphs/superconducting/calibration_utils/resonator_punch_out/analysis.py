import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from calibration_utils.error_codes import (
    ResonatorPunchOutErrorCode,
    ResonatorPunchOutCorrectiveAction,
)
# from qualibration_libs.analysis import peaks_dips  # Not used - now finding min values directly


# =========================
# Fit parameter container
# =========================

@dataclass
class FitParameters:
    success: bool
    resonator_frequency: float  # legacy: freq_shift + RF_frequency (kept for logging)
    frequency_shift: float
    optimal_power: float
    freq_low_abs: float = 0.0   # absolute resonator frequency at LOW power (use this to update state)
    error_code: int = ResonatorPunchOutErrorCode.SUCCESS  # Error diagnostic code
    corrective_action: int = ResonatorPunchOutCorrectiveAction.NONE  # Corrective action code
    action_magnitude: float = 0.0  # Magnitude of the corrective action


# =========================
# Logging
# =========================

def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for q in fit_results.keys():
        error_code = ResonatorPunchOutErrorCode(fit_results[q].get("error_code", 0))
        s_qubit = f"Results for qubit {q}: "
        s_power = f"Optimal readout power: {fit_results[q]['optimal_power']:.2f} dBm | "
        s_freq = f"Resonator frequency: {1e-9 * fit_results[q]['resonator_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.3f} MHz)\n"
        s_error = f"Error code: {error_code.name} ({error_code.value})\n"

        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"

        log_callable(s_qubit + s_error + s_power + s_freq + s_shift)


# =========================
# Raw data processing
# =========================

def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):

    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)

    full_freq = np.array(
        [ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]]
    )

    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}

    ds = ds.assign(
        IQ_abs_norm=ds.IQ_abs / ds.IQ_abs.mean(dim="detuning")
    )

    return ds


# =========================
# Shift-based fit logic
# =========================

def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:

    powers = ds.power.values
    if len(powers) != 2:
        raise ValueError(
            "Shift-based resonator analysis requires exactly 2 power points."
        )

    P_low, P_high = powers[0], powers[1]

    freq_low = []
    freq_high = []

    for q in node.namespace["qubits"]:
        name = q.name

        low_power_data = ds.sel(qubit=name, power=P_low).IQ_abs
        high_power_data = ds.sel(qubit=name, power=P_high).IQ_abs

        f0 = low_power_data.detuning[low_power_data.argmin(dim="detuning")].item()
        f1 = high_power_data.detuning[high_power_data.argmin(dim="detuning")].item()

        freq_low.append(f0)
        freq_high.append(f1)

    freq_low = xr.DataArray(freq_low, dims="qubit", coords={"qubit": ds.qubit})
    freq_high = xr.DataArray(freq_high, dims="qubit", coords={"qubit": ds.qubit})

    # Convention: freq_shift = freq_low - freq_high must be positive for punch-out
    # (Kerr effect shifts the resonator to lower frequencies at high power)
    freq_shift = freq_low - freq_high

    # Decision rule
    shift_threshold = node.parameters.frequency_shift_threshold_in_hz

    # Punch-out detected only when shift is positive and above threshold
    large_shift = freq_shift > shift_threshold

    # If shift is too large → punch-out detected → choose LOW power
    optimal_power = xr.where(
        large_shift,
        P_low,
        P_high,
    )


    ds_fit = ds.assign_coords(
        freq_shift=("qubit", freq_shift.data),
        freq_low=("qubit", freq_low.data),   # low-power resonance detuning (Hz from RF_frequency)
        optimal_power=("qubit", optimal_power.data),
    )

    return _extract_relevant_fit_parameters(ds_fit, node)


# =========================
# Result extraction
# =========================

def _extract_relevant_fit_parameters(
    fit: xr.Dataset, node: QualibrationNode
):
    """Extract fit parameters and determine success based on punch-out detection."""

    # Calculate absolute resonator frequencies from detunings and base RF frequency
    base_freq = np.array(
        [q.resonator.RF_frequency for q in node.namespace["qubits"]]
    )
    res_freq = fit.freq_shift + base_freq          # legacy: (freq_low - freq_high) + RF_freq
    freq_low_abs = fit.freq_low + base_freq        # actual low-power resonance frequency

    fit = fit.assign_coords(
        res_freq=("qubit", res_freq.data),
        freq_low_abs=("qubit", freq_low_abs.data),
    )
    fit.res_freq.attrs = {"long_name": "resonator frequency (legacy)", "units": "Hz"}
    fit.freq_low_abs.attrs = {"long_name": "low-power resonator frequency", "units": "Hz"}

    # Data validity checks
    no_nans = ~(np.isnan(fit.freq_shift.data) | np.isnan(fit.optimal_power.data))
    # Shift must be positive (freq_low - freq_high > 0) and within the swept span
    freq_in_range = (fit.freq_shift >= 0) & (fit.freq_shift < node.parameters.frequency_span_in_mhz * 1e6)
    wrong_direction = fit.freq_shift < 0  # Resonator shifted up at high power - unexpected

    # Punch-out detection: shift must be positive and above threshold
    shift_threshold = node.parameters.frequency_shift_threshold_in_hz
    punchout_detected = fit.freq_shift > shift_threshold

    # Success requires: valid data, positive shift in range, AND punch-out above threshold
    success = no_nans & freq_in_range & punchout_detected

    fit = fit.assign_coords(success=("qubit", success.data))

    # Build results dictionary with error codes
    fit_results = {}
    qubit_list = fit.qubit.values.tolist()
    for q in fit.qubit.values:
        q_idx = qubit_list.index(q)
        q_success = bool(fit.sel(qubit=q).success)
        q_shift = float(fit.freq_shift.sel(qubit=q))
        q_no_nans = bool(no_nans[q_idx])
        q_freq_in_range = bool(freq_in_range[q_idx])
        q_wrong_direction = bool(wrong_direction[q_idx])

        # Determine error code (checked in priority order)
        if q_success:
            error_code = ResonatorPunchOutErrorCode.SUCCESS
        elif not q_no_nans:
            error_code = ResonatorPunchOutErrorCode.INVALID_DATA
        elif q_wrong_direction:
            error_code = ResonatorPunchOutErrorCode.WRONG_SHIFT_DIRECTION
        elif not q_freq_in_range:
            error_code = ResonatorPunchOutErrorCode.INVALID_DATA
        elif q_shift < 1e3:  # Essentially zero shift
            error_code = ResonatorPunchOutErrorCode.NO_SHIFT_DETECTED
        else:
            error_code = ResonatorPunchOutErrorCode.SHIFT_BELOW_THRESHOLD

        fit_results[q] = FitParameters(
            success=q_success,
            resonator_frequency=float(fit.res_freq.sel(qubit=q)),
            frequency_shift=q_shift,
            optimal_power=float(fit.optimal_power.sel(qubit=q)),
            freq_low_abs=float(fit.freq_low_abs.sel(qubit=q)),
            error_code=int(error_code),
        )

    return fit, fit_results
