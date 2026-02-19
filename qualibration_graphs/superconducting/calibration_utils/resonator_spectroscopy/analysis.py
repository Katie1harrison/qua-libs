import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips
from calibration_utils.error_codes import (
    ResonatorSpectroscopyErrorCode,
    ResonatorSpectroscopyCorrectiveAction,
)


def _get_resonator_lo_frequency(qubit) -> Optional[float]:
    """Return the resonator LO (upconversion) frequency, or None if unavailable.

    Tries both the OPX+/Octave attribute and the LF/MW-FEM attribute.
    """
    try:
        return float(qubit.resonator.frequency_converter_up.LO_frequency)
    except AttributeError:
        pass
    try:
        return float(qubit.resonator.opx_output.upconverter_frequency)
    except AttributeError:
        return None


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit

    Amplitudes are extracted from the fitted Lorentzian curve, not raw data:
    - min_amplitude: baseline - amplitude (value at the dip)
    - max_amplitude: baseline (value away from resonance)
    """

    frequency: float
    fwhm: float
    min_amplitude: float
    max_amplitude: float
    success: bool
    error_code: int = ResonatorSpectroscopyErrorCode.SUCCESS
    corrective_action: int = ResonatorSpectroscopyCorrectiveAction.NONE
    action_magnitude: float = 0.0


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_freq = f"\tResonator frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_freq + s_fwhm)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit resonator spectroscopy data with a Lorentzian dip model.

    Extracts resonator frequency, linewidth, and min/max amplitudes from the fitted curve.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The QUAlibrate node.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    dict[str, FitParameters]
        Dictionary of fit parameters for each qubit.
    """
    # Fit the resonator line
    fit_results = peaks_dips(ds.IQ_abs, "detuning")
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results, ds, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(
    fit: xr.Dataset,
    ds: xr.Dataset,
    node: QualibrationNode,
):
    """Add metadata to the dataset and extract fit results."""

    # ------------------
    # Frequency metadata
    # ------------------
    fit.attrs = {"long_name": "frequency", "units": "Hz"}

    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + full_freq

    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    # -----
    # FWHM
    # -----
    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("qubit", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}

    # -------------------------------
    # Amplitude extraction from FIT (not raw data)
    # -------------------------------
    # For a Lorentzian dip:
    #   max_amplitude = baseline (away from resonance)
    #   min_amplitude = baseline - amplitude (at the dip)

    max_amp = fit.base_line.mean(dim="detuning")  # baseline is the max (away from resonance)
    min_amp = max_amp - fit.amplitude  # dip minimum

    fit = fit.assign_coords(
        min_amplitude=("qubit", min_amp.data),
        max_amplitude=("qubit", max_amp.data),
    )

    fit.min_amplitude.attrs = {"units": "V", "long_name": "min |IQ| (from fit)"}
    fit.max_amplitude.attrs = {"units": "V", "long_name": "max |IQ| (from fit)"}

    # ------------------
    # Success criteria
    # ------------------
    freq_success = (
        np.abs(res_freq.data)
        < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    )
    fwhm_success = (
        np.abs(fwhm.data)
        < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    )

    # SNR check: dip must be deeper than min_dip_contrast * baseline
    # This rejects noise spikes that peaks_dips may pick up when there is no real resonator dip
    position_found = ~np.isnan(fit.position.data)
    min_contrast = getattr(node.parameters, "min_dip_contrast", 0.15)
    baseline_mean = np.abs(max_amp.data)
    contrast = np.abs(baseline_mean - np.abs(fit.min_amplitude.data)) / baseline_mean
    contrast_success = contrast > min_contrast

    # LO leakage exclusion: discard peaks that land within ±lo_leakage_exclusion_mhz
    # of the resonator's LO upconversion frequency (these are LO artefacts, not real dips)
    lo_exclusion_mhz = getattr(node.parameters, "lo_leakage_exclusion_mhz", 10.0)
    lo_exclusion_hz = lo_exclusion_mhz * 1e6
    lo_not_too_close = np.ones(len(res_freq.data), dtype=bool)
    for i, q_obj in enumerate(node.namespace["qubits"]):
        lo_freq = _get_resonator_lo_frequency(q_obj)
        if lo_freq is not None and not np.isnan(res_freq.data[i]):
            lo_not_too_close[i] = abs(res_freq.data[i] - lo_freq) > lo_exclusion_hz

    success_criteria = freq_success & fwhm_success & position_found & contrast_success & lo_not_too_close
    fit = fit.assign_coords(success=("qubit", success_criteria))

    # Build results dictionary with error codes
    fit_results = {}
    qubit_list = fit.qubit.values.tolist()
    for q in fit.qubit.values:
        q_idx = qubit_list.index(q)
        q_success = bool(fit.sel(qubit=q).success.values)
        q_position_found = bool(position_found[q_idx])
        q_contrast_ok = bool(contrast_success[q_idx])

        if q_success:
            error_code = ResonatorSpectroscopyErrorCode.SUCCESS
        elif not q_position_found or not q_contrast_ok:
            error_code = ResonatorSpectroscopyErrorCode.NO_DIP_FOUND
        else:
            error_code = ResonatorSpectroscopyErrorCode.NO_DIP_FOUND

        fit_results[q] = FitParameters(
            frequency=fit.sel(qubit=q).res_freq.values.item(),
            fwhm=fit.sel(qubit=q).fwhm.values.item(),
            min_amplitude=fit.sel(qubit=q).min_amplitude.values.item(),
            max_amplitude=fit.sel(qubit=q).max_amplitude.values.item(),
            success=q_success,
            error_code=int(error_code),
        )

    return fit, fit_results
