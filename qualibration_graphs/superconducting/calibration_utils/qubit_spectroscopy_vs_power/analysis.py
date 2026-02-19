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
      - Compute |IQ| and phase (NO normalization)
      - Add full RF frequency coordinate
    """

    ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    ds = add_amplitude_and_phase(
        ds,
        dim="detuning",
        subtract_slope_flag=True,
    )

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


def _compute_fwhm_around_peak(detuning, signal, peak_idx) -> float:
    if peak_idx < 0:
        return np.nan

    x = np.asarray(detuning)
    y = np.asarray(signal)

    if np.all(np.isnan(y)):
        return np.nan

    y = y - np.nanmin(y)
    half_max = 0.5 * np.nanmax(y)

    above = y >= half_max
    if not np.any(above):
        return np.nan

    idx = np.where(above)[0]
    return x[idx[-1]] - x[idx[0]]


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


def _mask_blacklisted_detunings(ds: xr.Dataset, machine, tolerance_hz: float = 5e6) -> xr.Dataset:
    """
    Set IQ_abs to NaN at detuning points within ±tolerance_hz of any blacklisted
    qubit frequency. Applied per-qubit using the full_freq coordinate.

    This prevents blacklisted frequencies (e.g. those that yielded no Rabi oscillations
    in a previous attempt) from appearing as candidate peaks in subsequent retries.
    """
    if machine is None or not hasattr(machine, "temp_calibration") or machine.temp_calibration is None:
        return ds

    iq_masked = ds.IQ_abs.copy()

    for qubit_name in ds.qubit.values:
        try:
            bl_freqs = machine.temp_calibration[qubit_name].blacklisted_qubit_frequencies
        except (KeyError, TypeError, AttributeError):
            bl_freqs = None

        if not bl_freqs:
            continue

        # full_freq has dims (qubit, detuning) – absolute RF frequency in Hz
        full_freq_q = ds.full_freq.sel(qubit=qubit_name).values  # shape: (n_detuning,)

        blacklisted = np.zeros(len(full_freq_q), dtype=bool)
        for bl_freq in bl_freqs:
            blacklisted |= np.abs(full_freq_q - bl_freq) <= tolerance_hz

        if not np.any(blacklisted):
            continue

        # Zero out the blacklisted detuning points for this qubit across all powers
        keep = xr.DataArray(~blacklisted, dims=["detuning"], coords={"detuning": ds.detuning})
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
        # Fallback: use statistics from current data
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

    baseline_iq_abs_v = xr.DataArray(
        [_get_resonator_amplitude(machine, q, "min_amplitude", ds) for q in qubit_names],
        dims=["qubit"],
        coords={"qubit": qubit_names},
    )

    max_iq_abs_v = xr.DataArray(
        [_get_resonator_amplitude(machine, q, "max_amplitude", ds) for q in qubit_names],
        dims=["qubit"],
        coords={"qubit": qubit_names},
    )

    min_peak_height = p.min_peak_fraction * (max_iq_abs_v - baseline_iq_abs_v)

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
        & (ds.linewidth < p.linewidth_threshold_hz)
    )

    selected_power = (
        ds.power.where(valid_power)
        .max(dim="power")
        - p.power_buffer_db
    )

    ds["selected_power"] = selected_power

    def _peak_frequency(full_freq, iq_abs, power, target_power):
        if np.isnan(target_power):
            return np.nan

        diff = np.abs(power - target_power)
        if np.all(np.isnan(diff)):
            return np.nan

        idx = int(np.nanargmin(diff))
        spectrum = iq_abs[idx]

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

        # Check high baseline condition for all power points
        baseline_iq_abs = _get_resonator_amplitude(machine, q, "min_amplitude", ds)
        max_iq_abs = _get_resonator_amplitude(machine, q, "max_amplitude", ds)
        expected_peak_height = p.min_peak_fraction * (max_iq_abs - baseline_iq_abs)

        high_baseline_checks = []
        for power_idx in range(len(qubit_data.power)):
            signal = qubit_data.IQ_abs.isel(power=power_idx).values
            is_high_baseline = _check_high_baseline(
                signal,
                (max_iq_abs - baseline_iq_abs),
                p.linewidth_threshold_hz,
                detuning_step
            )
            high_baseline_checks.append(is_high_baseline)

        all_high_baseline = bool(all(high_baseline_checks))

        # Over-saturated if either condition is met (ensure Python bool, not numpy bool_)
        # over_saturated = bool(all_linewidths_large or all_high_baseline)
        over_saturated = bool(all_high_baseline)

        # Check for weak peak at high power (only if calibration failed)
        weak_peak_at_high_power = False
        weak_peak_frequency = np.nan
        weak_peak_power = np.nan
        if not (np.isfinite(qubit_data.selected_power) and np.isfinite(qubit_data.rough_qubit_frequency)):
            # Calibration failed, check if there's a weak peak at high power
            # Look at the highest 3 power points
            num_powers = len(qubit_data.power)
            high_power_indices = range(max(0, num_powers - 3), num_powers)

            # Weak peak is confirmed only if ALL three highest power points show a peak
            # above 50% of the expected height simultaneously (AND condition).
            all_weak = all(
                np.nanmax(qubit_data.IQ_abs.isel(power=pi).values) - baseline_iq_abs
                >= 0.8 * expected_peak_height
                for pi in high_power_indices
            )
            if all_weak:
                weak_peak_at_high_power = True
                # Report peak location from the highest power point (last index)
                signal = qubit_data.IQ_abs.isel(power=high_power_indices[-1]).values
                peak_idx = int(np.nanargmax(signal))
                # full_freq only has dimensions (qubit, detuning), not power
                weak_peak_frequency = float(qubit_data.full_freq.isel(detuning=peak_idx).values)
                weak_peak_power = float(qubit_data.power.isel(power=high_power_indices[-1]).values)

        # Determine error code based on detected conditions
        success = bool(
            np.isfinite(qubit_data.selected_power)
            and np.isfinite(qubit_data.rough_qubit_frequency)
        )

        if success and over_saturated:
            error_code = QubitSpectroscopyErrorCode.OVER_SATURATED_SUCCESS
        elif success:
            error_code = QubitSpectroscopyErrorCode.SUCCESS
        elif over_saturated:
            error_code = QubitSpectroscopyErrorCode.OVER_SATURATED
        elif weak_peak_at_high_power:
            error_code = QubitSpectroscopyErrorCode.WEAK_PEAK_HIGH_POWER
        else:
            error_code = QubitSpectroscopyErrorCode.NO_PEAK_FOUND

        fit_results[q] = FitParameters(
            selected_power=qubit_data.selected_power.values.__float__(),
            rough_qubit_frequency=qubit_data.rough_qubit_frequency.values.__float__(),
            linewidth=qubit_data.linewidth.min(dim="power").values.__float__(),
            success=success,
            over_saturated=over_saturated,
            weak_peak_at_high_power=weak_peak_at_high_power,
            weak_peak_frequency=weak_peak_frequency,
            weak_peak_power=weak_peak_power,
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
            if weak_peak_at_high_power:
                weak_freq = result.get('weak_peak_frequency', np.nan)
                if np.isfinite(weak_freq):
                    log_callable(
                        f"[{qubit_name}] FAILED - Error code: {error_code.name} ({error_code.value})\n"
                        f"  Weak peak detected at high power\n"
                        f"  Weak peak frequency: {weak_freq / 1e9:.6f} GHz"
                    )
                else:
                    log_callable(
                        f"[{qubit_name}] FAILED - Error code: {error_code.name} ({error_code.value})\n"
                        f"  Weak peak detected at high power"
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
    weak_peak_at_high_power: bool = False  # True if weak peak appears only at high power
    weak_peak_frequency: float = np.nan  # Frequency of the weak peak (if detected)
    weak_peak_power: float = np.nan  # Drive power at which the weak peak was detected (dBm)
    error_code: int = QubitSpectroscopyErrorCode.SUCCESS  # Error diagnostic code
    corrective_action: int = QubitSpectroscopyCorrectiveAction.NONE  # Corrective action code
    action_magnitude: float = 0.0  # Magnitude of the corrective action
