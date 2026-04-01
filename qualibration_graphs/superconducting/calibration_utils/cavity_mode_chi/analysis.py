"""
Analysis utilities for the cavity mode chi (dispersive shift) measurement.

The sequence performs two interleaved qubit spectroscopy sweeps:
  - Branch 0 (no photon): qubit ge spectroscopy in the |g,0> state.
  - Branch 1 (photon):    qubit ge spectroscopy after Fock1 preparation (|g,1> state).

The qubit ge transition shifts by chi when there is one photon in the cavity
(chi = f_ge_0 - f_ge_1). Fitting two Lorentzian peaks gives chi.

State update:
    - cavity_mode.chi = chi_hz  [Hz]
"""
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Fit results for a single qubit's chi measurement."""
    chi_hz: float
    """Dispersive shift chi = f_ge(0 photons) - f_ge(1 photon) [Hz]."""
    f_ge_0_hz: float
    """Qubit ge frequency with 0 photons in the cavity [Hz offset from center]."""
    f_ge_1_hz: float
    """Qubit ge frequency with 1 photon in the cavity [Hz offset from center]."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, result in fit_results.items():
        status = "SUCCESS" if result["success"] else "FAIL"
        chi_khz = result["chi_hz"] * 1e-3
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tchi = {chi_khz:.3f} kHz | "
            f"f_ge_0 offset = {result['f_ge_0_hz'] * 1e-3:.3f} kHz | "
            f"f_ge_1 offset = {result['f_ge_1_hz'] * 1e-3:.3f} kHz"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert I/Q to Volts if not using state discrimination."""
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def _lorentzian(x, a, x0, gamma, offset):
    return offset + a / (1 + ((x - x0) / gamma) ** 2)


def _fit_lorentzian_peak(x: np.ndarray, y: np.ndarray):
    """Fit a Lorentzian peak; returns (x0, success)."""
    try:
        idx_peak = np.argmax(y)
        a0 = float(y[idx_peak] - np.mean(y))
        x0_0 = float(x[idx_peak])
        gamma0 = float((x[-1] - x[0]) / 10)
        offset0 = float(np.mean(y))
        popt, _ = curve_fit(
            _lorentzian, x, y,
            p0=[a0, x0_0, gamma0, offset0],
            maxfev=10000,
        )
        return float(popt[1]), True
    except Exception:
        return float("nan"), False


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit two Lorentzian peaks (0-photon and 1-photon branches) and extract chi."""
    signal_name = "state" if node.parameters.use_state_discrimination else "I"

    fit_results = {}
    fit_data_vars = {}

    for q in ds.qubit.values:
        ds_q = ds.sel(qubit=q)
        x = ds_q.detuning.values  # in Hz

        # Branch 0: no photon
        y0 = getattr(ds_q, signal_name).sel(photon_branch=0).values
        f0, ok0 = _fit_lorentzian_peak(x, y0)

        # Branch 1: one photon
        y1 = getattr(ds_q, signal_name).sel(photon_branch=1).values
        f1, ok1 = _fit_lorentzian_peak(x, y1)

        if ok0 and ok1 and np.isfinite(f0) and np.isfinite(f1):
            chi_hz = float(f0 - f1)
            success = True
        else:
            chi_hz = float("nan")
            success = False

        fit_results[q] = FitParameters(
            chi_hz=chi_hz,
            f_ge_0_hz=f0,
            f_ge_1_hz=f1,
            success=success,
        )

    return ds, fit_results
