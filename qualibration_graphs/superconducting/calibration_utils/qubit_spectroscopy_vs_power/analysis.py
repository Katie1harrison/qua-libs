import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Dict, Tuple

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from calibration_utils.error_codes import (
    QubitSpectroscopyErrorCode,
    QubitSpectroscopyCorrectiveAction,
)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process raw qubit spectroscopy vs power dataset:
      - Convert I/Q to Volts
      - Add full RF frequency coordinate
    """

    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)

    full_freq = np.array(
        [ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]]
    )

    ds = ds.assign_coords(
        full_freq=(["qubit", "detuning"], full_freq)
    )

    ds.full_freq.attrs = {
        "long_name": "RF frequency",
        "units": "Hz",
    }

    return ds


def _peak_index(iq_abs, baseline, min_height):
    y = np.asarray(iq_abs)

    if np.all(np.isnan(y)):
        return -1

    idx = int(np.nanargmax(y))
    if y[idx] - baseline < min_height:
        return -1

    return idx


def _apply_persistence_filter(
    peak_indices: np.ndarray,
    detuning: np.ndarray,
    lookahead: int,
    freq_tolerance_hz: float,
) -> np.ndarray:
    """
    Remove isolated peaks that do not persist at higher power levels.

    A peak at power index i is discarded (set to -1) when none of the next
    ``lookahead`` higher power levels contain a peak within
    ``freq_tolerance_hz`` of it.  Power levels at the very end of the sweep
    (where fewer than ``lookahead`` higher levels exist) are kept as-is —
    their persistence cannot be assessed.

    Parameters
    ----------
    peak_indices : (n_power,) int array
        Index of the detected peak in the detuning axis for each power level.
        -1 means no peak was found at that power.
    detuning : (n_detuning,) float array
        Detuning axis values in Hz.
    lookahead : int
        Number of higher power levels to inspect for a matching peak.
    freq_tolerance_hz : float
        Maximum allowed frequency separation (Hz) for two peaks to be
        considered the same transition.
    """
    n_power = len(peak_indices)
    filtered = peak_indices.copy()

    for i in range(n_power):
        if peak_indices[i] < 0:
            continue  # no peak here — nothing to filter

        n_higher = n_power - i - 1
        if n_higher == 0:
            continue  # no higher powers to check — keep unconditionally

        n_to_check = min(lookahead, n_higher)
        freq_i = detuning[peak_indices[i]]

        found = False
        for j in range(i + 1, i + 1 + n_to_check):
            if peak_indices[j] >= 0:
                if abs(detuning[peak_indices[j]] - freq_i) <= freq_tolerance_hz:
                    found = True
                    break

        if not found:
            filtered[i] = -1

    return filtered


def _compute_fwhm_around_peak(detuning, signal, peak_idx) -> float:
    """
    Compute FWHM of the peak at peak_idx using linear interpolation at the
    half-max crossings, giving sub-step accuracy even on coarse sweeps.
    """
    if peak_idx < 0:
        return np.nan

    x = np.asarray(detuning, dtype=float)
    y = np.asarray(signal, dtype=float)

    if np.all(np.isnan(y)):
        return np.nan

    y = y - np.nanmin(y)
    half_max = 0.5 * float(np.nanmax(y))

    above = y >= half_max
    if not np.any(above):
        return np.nan

    idx = np.where(above)[0]
    left_i = int(idx[0])
    right_i = int(idx[-1])

    # Interpolate left crossing between (left_i - 1) and left_i
    if left_i > 0 and not above[left_i - 1]:
        dy = y[left_i] - y[left_i - 1]
        left_x = (
            x[left_i - 1] + (half_max - y[left_i - 1]) / dy * (x[left_i] - x[left_i - 1])
            if dy > 0 else x[left_i]
        )
    else:
        left_x = x[left_i]  # peak extends to sweep edge

    # Interpolate right crossing between right_i and (right_i + 1)
    if right_i < len(x) - 1 and not above[right_i + 1]:
        dy = y[right_i + 1] - y[right_i]
        right_x = (
            x[right_i] + (half_max - y[right_i]) / dy * (x[right_i + 1] - x[right_i])
            if dy < 0 else x[right_i]
        )
    else:
        right_x = x[right_i]  # peak extends to sweep edge

    return right_x - left_x


def _check_high_baseline(signal, min_peak_height, linewidth_threshold_hz, detuning_step) -> bool:
    """
    Check if the mean signal is consistently high (>20% of expected peak)
    for an interval larger than 10 times the linewidth threshold.

    Returns True if over-saturated (high baseline).
    """
    y = np.asarray(signal)

    if np.all(np.isnan(y)):
        return False

    # Compute baseline (minimum) and threshold
    baseline = np.nanmin(y)
    threshold = baseline + 0.2 * min_peak_height

    # Find where signal exceeds threshold
    above_threshold = y >= threshold

    if not np.any(above_threshold):
        return False

    # Find longest continuous interval above threshold
    # Use run-length encoding
    changes = np.diff(np.concatenate([[False], above_threshold, [False]]).astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    if len(starts) == 0:
        return False

    # Compute interval widths in Hz
    max_interval_points = np.max(ends - starts)
    max_interval_hz = max_interval_points * detuning_step

    # Check if any interval exceeds 10× linewidth threshold (convert to Python bool)
    return bool(max_interval_hz > 10 * linewidth_threshold_hz)


def _mask_blacklisted_detunings(
    ds: xr.Dataset,
    machine,
    freq_tolerance_hz: float = 5e6,
    power_tolerance_dbm: float = 3.0,
) -> xr.Dataset:
    """
    Set IQ_abs to NaN at 2-D (frequency, power) regions near any blacklisted
    (qubit_freq_hz, drive_power_dbm) pair. The exclusion zone is:

        |freq  - bl_freq|  ≤ freq_tolerance_hz   AND
        |power - bl_power| ≤ power_tolerance_dbm

    This lets a subsequent spectroscopy run detect the same qubit transition at a
    *different* drive power, rather than discarding the frequency column entirely.
    """
    if machine is None or not hasattr(machine, "temp_calibration") or machine.temp_calibration is None:
        return ds

    iq_masked = ds.IQ_abs.copy()
    power_vals = ds.power.values  # shape: (n_power,)

    for qubit_name in ds.qubit.values:
        try:
            bl_points = machine.temp_calibration[qubit_name].blacklisted_qubit_points
        except (KeyError, TypeError, AttributeError):
            bl_points = None

        if not bl_points:
            continue

        # full_freq has dims (qubit, detuning) – absolute RF frequency in Hz
        full_freq_q = ds.full_freq.sel(qubit=qubit_name).values  # shape: (n_detuning,)

        for bl_freq, bl_power in bl_points:
            freq_mask = np.abs(full_freq_q - bl_freq) <= freq_tolerance_hz   # (n_detuning,)
            power_mask = np.abs(power_vals - bl_power) <= power_tolerance_dbm  # (n_power,)

            if not np.any(freq_mask) or not np.any(power_mask):
                continue

            # Outer product → True where BOTH conditions hold: shape (n_power, n_detuning)
            to_null = np.outer(power_mask, freq_mask)
            keep = xr.DataArray(
                ~to_null,
                dims=["power", "detuning"],
                coords={"power": ds.power, "detuning": ds.detuning},
            )
            iq_masked.loc[dict(qubit=qubit_name)] = iq_masked.sel(qubit=qubit_name).where(keep)

    return ds.assign(IQ_abs=iq_masked)


def _get_resonator_amplitude(machine, qubit_name: str, key: str, data: xr.Dataset) -> float:
    """
    Get resonator amplitude from temp_calibration with fallback to data statistics.

    If resonator_spectroscopy hasn't been run, use min/max from the current dataset.
    """
    try:
        return machine.temp_calibration[qubit_name].resonator_amplitudes[key]
    except (KeyError, TypeError, AttributeError):
        # Fallback: use statistics from current data (fallback: IQ_abs.min/max)
        qubit_data = data.sel(qubit=qubit_name).IQ_abs
        if key == "min_amplitude":
            return float(qubit_data.min())
        elif key == "max_amplitude":
            return float(qubit_data.max())
        else:
            raise ValueError(f"Unknown amplitude key: {key}")


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[xr.Dataset, Dict[str, "FitParameters"]]:
    """
    Peak-based rough qubit spectroscopy vs power analysis.
    """

    p = node.parameters
    machine = node.machine

    # Mask out detuning points near blacklisted qubit frequencies before any peak finding
    ds = _mask_blacklisted_detunings(ds, machine)

    qubit_names = ds.qubit.values

    # Current-data statistics: adaptive to the actual signal scale in this sweep.
    baseline_iq_abs_v = ds.IQ_abs.min(dim=["power", "detuning"])
    max_iq_abs_v = ds.IQ_abs.max(dim=["power", "detuning"])
    data_range_v = max_iq_abs_v - baseline_iq_abs_v

    # Resonator-amplitude-based range: the expected IQ dynamic range of the readout
    # system, calibrated during resonator spectroscopy.  Falls back to data statistics
    # if resonator calibration has not been run yet.
    resonator_baseline_v = xr.DataArray(
        [_get_resonator_amplitude(machine, q, "min_amplitude", ds) for q in qubit_names],
        dims=["qubit"], coords={"qubit": qubit_names},
    )
    resonator_max_v = xr.DataArray(
        [_get_resonator_amplitude(machine, q, "max_amplitude", ds) for q in qubit_names],
        dims=["qubit"], coords={"qubit": qubit_names},
    )
    resonator_range_v = resonator_max_v - resonator_baseline_v

    # Use the more stringent (larger) range as the height reference so that a peak
    # must be a meaningful fraction of the full IQ dynamic range, not just a fraction
    # of whatever happened to appear in a low-SNR sweep.
    height_range_v = xr.where(data_range_v >= resonator_range_v, data_range_v, resonator_range_v)

    min_peak_height = p.min_peak_fraction * height_range_v

    peak_index = xr.apply_ufunc(
        _peak_index,
        ds.IQ_abs,
        baseline_iq_abs_v,
        min_peak_height,
        input_core_dims=[["detuning"], [], []],
        vectorize=True,
        output_dtypes=[int],
    )

    ds["peak_index"] = peak_index

    # Persistence filter: discard peaks that do not reappear at any of the
    # next `peak_persistence_lookahead` higher-power levels within the
    # allowed frequency tolerance.  Such isolated peaks are most likely noise
    # artefacts rather than the real qubit transition.
    if int(p.peak_persistence_lookahead) > 0:
        peak_index = xr.apply_ufunc(
            _apply_persistence_filter,
            peak_index,
            ds.detuning,
            kwargs={
                "lookahead": int(p.peak_persistence_lookahead),
                "freq_tolerance_hz": float(p.peak_persistence_freq_tolerance_hz),
            },
            input_core_dims=[["power"], ["detuning"]],
            output_core_dims=[["power"]],
            vectorize=True,
            output_dtypes=[int],
        )
        ds["peak_index"] = peak_index

    ds["peak_height"] = xr.where(
        peak_index >= 0,
        ds.IQ_abs.isel(detuning=peak_index),
        np.nan,
    )

    linewidth = xr.apply_ufunc(
        _compute_fwhm_around_peak,
        ds.detuning,
        ds.IQ_abs,
        peak_index,
        input_core_dims=[["detuning"], ["detuning"], []],
        vectorize=True,
        output_dtypes=[float],
    )

    ds["linewidth"] = linewidth

    valid_power = (
        (ds.peak_index >= 0)
        & (ds.linewidth <= p.linewidth_threshold_hz)
    )

    # Primary: among powers where peak found AND linewidth ≤ threshold, select
    # the one with maximum linewidth (highest power before over-saturation).
    # idxmax returns the power coordinate value at the maximum, NaN if all are NaN.
    primary_selected = (
        ds.linewidth.where(valid_power)
        .idxmax(dim="power", skipna=True)
    )

    # Fallback: if the linewidth threshold is never met (e.g. all detected peaks
    # are power-broadened), fall back to the power with the narrowest detected
    # linewidth regardless of the threshold.  This is flagged as OVER_SATURATED_SUCCESS.
    fallback_selected = (
        ds.linewidth.where(ds.peak_index >= 0)
        .idxmin(dim="power", skipna=True)
    )

    used_fallback = ~np.isfinite(primary_selected)
    selected_power = (
        primary_selected.where(~used_fallback, other=fallback_selected)
        - p.power_buffer_db
    )

    ds["selected_power"] = selected_power
    ds["used_fallback_power"] = used_fallback

    def _peak_frequency(full_freq, i_data, power, target_power):
        if np.isnan(target_power):
            return np.nan

        diff = np.abs(power - target_power)
        if np.all(np.isnan(diff)):
            return np.nan

        idx = int(np.nanargmin(diff))
        spectrum = i_data[idx]

        if np.all(np.isnan(spectrum)):
            return np.nan

        peak_idx = int(np.nanargmax(spectrum))
        return full_freq[peak_idx]

    rough_freq = xr.apply_ufunc(
        _peak_frequency,
        ds.full_freq,
        ds.IQ_abs,
        ds.power,
        ds.selected_power,
        input_core_dims=[
            ["detuning"],
            ["power", "detuning"],
            ["power"],
            [],
        ],
        vectorize=True,
        output_dtypes=[float],
    )

    ds["rough_qubit_frequency"] = rough_freq

    # Detect over-saturation for each qubit
    detuning_step = float(np.diff(ds.detuning.values).mean())

    fit_results = {}
    for q in ds.qubit.values:
        qubit_data = ds.sel(qubit=q)

        # Check saturation conditions across all power points
        all_linewidths_large = bool(np.all(
            qubit_data.linewidth.values >= p.linewidth_threshold_hz
        ) or np.all(np.isnan(qubit_data.linewidth.values)))

        # Compute height reference: max of data-derived range and resonator-amplitude range.
        baseline_iq_abs = float(qubit_data.IQ_abs.min())
        max_iq_abs = float(qubit_data.IQ_abs.max())
        data_range = max_iq_abs - baseline_iq_abs
        resonator_range = (
            _get_resonator_amplitude(machine, q, "max_amplitude", ds)
            - _get_resonator_amplitude(machine, q, "min_amplitude", ds)
        )
        height_range = max(data_range, resonator_range)

        high_baseline_checks = []
        for power_idx in range(len(qubit_data.power)):
            signal = qubit_data.IQ_abs.isel(power=power_idx).values
            is_high_baseline = _check_high_baseline(
                signal,
                height_range,
                p.linewidth_threshold_hz,
                detuning_step
            )
            high_baseline_checks.append(is_high_baseline)

        all_high_baseline = bool(all(high_baseline_checks))

        # Over-saturated if either condition is met (ensure Python bool, not numpy bool_)
        # over_saturated = bool(all_linewidths_large or all_high_baseline)
        over_saturated = bool(all_high_baseline)

        # Determine error code based on detected conditions
        success = bool(
            np.isfinite(qubit_data.selected_power)
            and np.isfinite(qubit_data.rough_qubit_frequency)
        )

        used_fallback_q = bool(ds["used_fallback_power"].sel(qubit=q).item())

        if success and (over_saturated or used_fallback_q):
            error_code = QubitSpectroscopyErrorCode.OVER_SATURATED_SUCCESS
        elif success:
            error_code = QubitSpectroscopyErrorCode.SUCCESS
        elif over_saturated:
            error_code = QubitSpectroscopyErrorCode.OVER_SATURATED
        else:
            error_code = QubitSpectroscopyErrorCode.NO_PEAK_FOUND

        fit_results[q] = FitParameters(
            selected_power=qubit_data.selected_power.values.__float__(),
            rough_qubit_frequency=qubit_data.rough_qubit_frequency.values.__float__(),
            linewidth=qubit_data.linewidth.min(dim="power").values.__float__(),
            success=success,
            over_saturated=over_saturated,
            error_code=int(error_code),
        )

    return ds, fit_results


def log_fitted_results(fit_results: Dict[str, Dict], log_callable=None):
    """Log the fitted results for each qubit."""
    if log_callable is None:
        log_callable = print

    for qubit_name, result in fit_results.items():
        success = result.get("success", False)
        over_saturated = result.get("over_saturated", False)
        weak_peak_at_high_power = result.get("weak_peak_at_high_power", False)
        error_code = QubitSpectroscopyErrorCode(result.get("error_code", 0))

        if success:
            status = "SUCCESS"
            if over_saturated:
                status += " (OVER-SATURATED)"
            log_callable(
                f"[{qubit_name}] {status} - Error code: {error_code.name} ({error_code.value})\n"
                f"  Selected power: {result['selected_power']:.2f} dBm\n"
                f"  Qubit frequency: {result['rough_qubit_frequency'] / 1e9:.6f} GHz\n"
                f"  Min linewidth: {result['linewidth'] / 1e6:.2f} MHz"
            )
        else:
            log_callable(
                f"[{qubit_name}] FAILED - Error code: {error_code.name} ({error_code.value})\n"
                f"  No valid peak found"
            )


# ----------------------------------------------------------------------
# Fit interface
# ----------------------------------------------------------------------

@dataclass
class FitParameters:
    """Spectroscopy vs power fit results for a single qubit"""

    selected_power: float
    rough_qubit_frequency: float
    linewidth: float
    success: bool
    over_saturated: bool = False  # True if all power points show over-saturation
    error_code: int = QubitSpectroscopyErrorCode.SUCCESS  # Error diagnostic code
    corrective_action: int = QubitSpectroscopyCorrectiveAction.NONE  # Corrective action code
    action_magnitude: float = 0.0  # Magnitude of the corrective action
