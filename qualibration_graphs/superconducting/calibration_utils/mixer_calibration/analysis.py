import logging
from dataclasses import dataclass
from typing import Dict

from qualibrate import QualibrationNode
from qualang_tools.octave_tools.calibration_result_plotter import CalibrationResultPlotter


@dataclass
class FitParameters:
    """Stores the mixer calibration results for a single qubit or cavity mode"""

    resonator: dict
    xy_drive: dict
    success: bool
    cavity_mode_drive: dict = None
    sideband_drive: dict = None


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
        s_qubit = f"Results for {q}: "
        if fit_results[q]["resonator"] is not None:
            s_res = f"\tresonator         -> LO leakage suppression: {fit_results[q]['resonator']['lo_leakage']:.1f} dB | image rejection: {fit_results[q]['resonator']['image_rejection']:.1f} dB.\n"
        else:
            s_res = ""
        if fit_results[q]["xy_drive"] is not None:
            s_xy = f"\txy_drive          -> LO leakage suppression: {fit_results[q]['xy_drive']['lo_leakage']:.1f} dB | image rejection: {fit_results[q]['xy_drive']['image_rejection']:.1f} dB.\n"
        else:
            s_xy = ""
        if fit_results[q].get("cavity_mode_drive") is not None:
            s_cav = f"\tcavity_mode_drive -> LO leakage suppression: {fit_results[q]['cavity_mode_drive']['lo_leakage']:.1f} dB | image rejection: {fit_results[q]['cavity_mode_drive']['image_rejection']:.1f} dB.\n"
        else:
            s_cav = ""
        if fit_results[q].get("sideband_drive") is not None:
            s_f0g1 = f"\tsideband_drive        -> LO leakage suppression: {fit_results[q]['sideband_drive']['lo_leakage']:.1f} dB | image rejection: {fit_results[q]['sideband_drive']['image_rejection']:.1f} dB.\n"
        else:
            s_f0g1 = ""
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_res + s_xy + s_cav + s_f0g1)


def extract_relevant_fit_parameters(node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    cal_results = node.namespace["calibration_results"]
    fit_results = {}
    for q, cal in cal_results.items():
        resonator = (
            {
                "lo_leakage": CalibrationResultPlotter(cal["resonator"]).get_lo_leakage_rejection(),
                "image_rejection": CalibrationResultPlotter(cal["resonator"]).get_image_rejection(),
            }
            if node.parameters.calibrate_resonator and cal.get("resonator") is not None
            else None
        )
        xy_drive = (
            {
                "lo_leakage": CalibrationResultPlotter(cal["xy_drive"]).get_lo_leakage_rejection(),
                "image_rejection": CalibrationResultPlotter(cal["xy_drive"]).get_image_rejection(),
            }
            if node.parameters.calibrate_drive and cal.get("xy_drive") is not None
            else None
        )
        cavity_mode_drive = (
            {
                "lo_leakage": CalibrationResultPlotter(cal["cavity_mode_drive"]).get_lo_leakage_rejection(),
                "image_rejection": CalibrationResultPlotter(cal["cavity_mode_drive"]).get_image_rejection(),
            }
            if node.parameters.calibrate_cavity_drive and cal.get("cavity_mode_drive") is not None
            else None
        )
        sideband_drive = (
            {
                "lo_leakage": CalibrationResultPlotter(cal["sideband_drive"]).get_lo_leakage_rejection(),
                "image_rejection": CalibrationResultPlotter(cal["sideband_drive"]).get_image_rejection(),
            }
            if node.parameters.calibrate_sideband_drive and cal.get("sideband_drive") is not None
            else None
        )
        fit_results[q] = FitParameters(
            resonator=resonator,
            xy_drive=xy_drive,
            cavity_mode_drive=cavity_mode_drive,
            sideband_drive=sideband_drive,
            success=True,
        )
    return fit_results
