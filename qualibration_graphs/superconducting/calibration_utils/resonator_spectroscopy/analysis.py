import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips
from calibration_utils.error_codes import (
    ResonatorSpectroscopyErrorCode,
    ResonatorSpectroscopyCorrectiveAction,
)


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit."""

    frequency: float
    fwhm: float
    success: bool
    error_code: int = ResonatorSpectroscopyErrorCode.SUCCESS
    corrective_action: int = ResonatorSpectroscopyCorrectiveAction.NONE
    action_magnitude: float = 0.0


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the fitted results for all qubits."""
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
    """Fit the resonator spectroscopy dip for each qubit."""
    fit_results = peaks_dips(ds.IQ_abs, "detuning")
    fit_data, fit_results = _extract_relevant_fit_parameters(fit_results, ds, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, ds: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and extract fit results."""
    fit.attrs = {"long_name": "frequency", "units": "Hz"}

    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    res_freq = fit.position + full_freq
    fit = fit.assign_coords(res_freq=("qubit", res_freq.data))
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}

    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=("qubit", fwhm.data))
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}

    freq_success = np.abs(res_freq.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    fwhm_success = np.abs(fwhm.data) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq
    position_found = ~np.isnan(fit.position.data)

    success_criteria = freq_success & fwhm_success & position_found
    fit = fit.assign_coords(success=("qubit", success_criteria))

    fit_results = {}
    for q in fit.qubit.values:
        q_success = bool(fit.sel(qubit=q).success.values)
        error_code = (
            ResonatorSpectroscopyErrorCode.SUCCESS
            if q_success
            else ResonatorSpectroscopyErrorCode.NO_DIP_FOUND
        )
        fit_results[q] = FitParameters(
            frequency=fit.sel(qubit=q).res_freq.values.item(),
            fwhm=fit.sel(qubit=q).fwhm.values.item(),
            success=q_success,
            error_code=int(error_code),
        )

    return fit, fit_results
