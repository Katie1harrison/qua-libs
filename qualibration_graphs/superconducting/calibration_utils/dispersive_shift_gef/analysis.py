"""
Analysis utilities for the GEF dispersive shift measurement.

Three resonator spectra are acquired:
  - state 0 (|g⟩): qubit thermalized, no drive
  - state 1 (|e⟩): qubit driven to |e⟩ via x180_ge
  - state 2 (|f⟩): qubit driven to |f⟩ via x180_ge + x180_ef

The dispersive shifts:
  chi_ge = f_resonator(|e⟩) - f_resonator(|g⟩)
  chi_ef = f_resonator(|f⟩) - f_resonator(|e⟩)

The optimal readout frequency is set to f_resonator(|e⟩), which maximises
contrast between |g⟩ and |e⟩ while giving good sensitivity for the |f⟩ state.

All spectra are fitted with Lorentzian dip models.
"""
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis.models import lorentzian_dip


@dataclass
class FitParametersGEF:
    """Fit results for the GEF dispersive shift measurement."""
    f_g: float
    """Resonator frequency when qubit is in |g⟩ [Hz]."""
    f_e: float
    """Resonator frequency when qubit is in |e⟩ [Hz]."""
    f_f: float
    """Resonator frequency when qubit is in |f⟩ [Hz]."""
    chi_ge_hz: float
    """GE dispersive shift chi_ge = f_e - f_g [Hz]."""
    chi_ef_hz: float
    """EF dispersive shift chi_ef = f_f - f_e [Hz]."""
    f_optimal: float
    """Optimal readout frequency = f_resonator(|e⟩) [Hz]."""
    success: bool
    # Lorentzian fit parameters (default nan = fit not attempted / failed)
    kappa_g_hz: float = float("nan")
    """Half-linewidth (HWHM) of |g⟩ resonator dip [Hz]."""
    kappa_e_hz: float = float("nan")
    """Half-linewidth (HWHM) of |e⟩ resonator dip [Hz]."""
    kappa_f_hz: float = float("nan")
    """Half-linewidth (HWHM) of |f⟩ resonator dip [Hz]."""
    amplitude_g: float = float("nan")
    amplitude_e: float = float("nan")
    amplitude_f: float = float("nan")
    offset_g: float = float("nan")
    offset_e: float = float("nan")
    offset_f: float = float("nan")
    center_g_hz: float = float("nan")
    """Fitted dip center detuning for |g⟩ [Hz]."""
    center_e_hz: float = float("nan")
    """Fitted dip center detuning for |e⟩ [Hz]."""
    center_f_hz: float = float("nan")
    """Fitted dip center detuning for |f⟩ [Hz]."""
    # Phase fit: φ(ω) = φ₀ − arctan(2·(ω − ωr) / κ)
    phi0_g: float = float("nan")
    phi0_e: float = float("nan")
    phi0_f: float = float("nan")
    kappa_phase_g_hz: float = float("nan")
    kappa_phase_e_hz: float = float("nan")
    kappa_phase_f_hz: float = float("nan")
    center_phase_g_hz: float = float("nan")
    center_phase_e_hz: float = float("nan")
    center_phase_f_hz: float = float("nan")


def _fit_phase(x: np.ndarray, phase: np.ndarray, center0: float, kappa0: float):
    """Fit resonator phase: φ(ω) = φ₀ − arctan(2·(ω − ωr) / κ)

    Returns (phi0, center_hz, kappa_hz, success).
    """
    x = np.asarray(x, dtype=float)
    phase = np.asarray(phase, dtype=float)
    phi0_0 = float(np.mean(phase))
    center0 = float(center0)
    kappa0 = max(float(kappa0), float(x[1] - x[0]))

    def _model(xv, phi0, center, kappa):
        return phi0 - np.arctan(2.0 * (xv - center) / kappa)

    try:
        popt, _ = curve_fit(
            _model, x, phase,
            p0=[phi0_0, center0, kappa0],
            bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
        phi0, center, kappa = popt
        return float(phi0), float(center), float(abs(kappa)), True
    except Exception:
        return phi0_0, center0, kappa0, False


def _fit_lorentzian_dip(x: np.ndarray, y: np.ndarray):
    """Fit a Lorentzian dip to 1D data.

    Returns (center_hz, kappa_hz, amplitude, offset, success) where kappa is HWHM.
    Model: y = offset - amplitude * kappa^2 / (kappa^2 + (x - center)^2)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    offset0 = float(np.max(y))
    i_min = int(np.argmin(y))
    center0 = float(x[i_min])
    amplitude0 = float(offset0 - np.min(y))
    amplitude0 = max(amplitude0, 1e-12)

    half_depth = offset0 - 0.5 * amplitude0
    try:
        left_idx = np.where(y[:i_min + 1] > half_depth)[0]
        right_idx = np.where(y[i_min:] > half_depth)[0]
        left = float(x[left_idx[-1]]) if len(left_idx) else float(x[0])
        right = float(x[i_min + right_idx[0]]) if len(right_idx) else float(x[-1])
        kappa0 = max(0.5 * abs(right - left), float(x[1] - x[0]))
    except (IndexError, ValueError):
        kappa0 = float((x[-1] - x[0]) / 8)

    try:
        popt, _ = curve_fit(
            lorentzian_dip, x, y,
            p0=[amplitude0, center0, kappa0, offset0],
            bounds=([0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
        amplitude, center, kappa, offset = popt
        return float(center), float(abs(kappa)), float(amplitude), float(offset), True
    except Exception:
        return center0, kappa0, amplitude0, offset0, False


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"Results for qubit {q}: {status}\n"
            f"\tf_g: {1e-9 * res['f_g']:.5f} GHz "
            f"(kappa_g FWHM: {2e-3 * res.get('kappa_g_hz', float('nan')):.1f} kHz)\n"
            f"\tf_e: {1e-9 * res['f_e']:.5f} GHz "
            f"(kappa_e FWHM: {2e-3 * res.get('kappa_e_hz', float('nan')):.1f} kHz)\n"
            f"\tf_f: {1e-9 * res['f_f']:.5f} GHz "
            f"(kappa_f FWHM: {2e-3 * res.get('kappa_f_hz', float('nan')):.1f} kHz)\n"
            f"\tchi_ge: {1e-3 * res['chi_ge_hz']:.1f} kHz | "
            f"chi_ef: {1e-3 * res['chi_ef_hz']:.1f} kHz | "
            f"f_opt (|e>): {1e-9 * res['f_optimal']:.5f} GHz"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Convert raw I/Q to Volts and add a full_freq coordinate."""
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array(
        [ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]]
    )
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParametersGEF]]:
    """Fit |g⟩, |e⟩, and |f⟩ resonator spectra with Lorentzians and extract chi.

    Expects ``ds`` to have a ``qubit_state`` dimension with values ``[0, 1, 2]``
    (0 = ground, 1 = excited, 2 = f-state).

    Returns
    -------
    ds : xr.Dataset  (unchanged — fit curves reconstructed in plotting)
    fit_results : dict[str, FitParametersGEF]
    """
    span_hz = node.parameters.frequency_span_in_mhz * 1e6
    fit_results = {}

    for q_obj in node.namespace["qubits"]:
        q = q_obj.name
        rr_freq = q_obj.resonator.RF_frequency
        dfs = ds.detuning.values.astype(float)

        amp_g = ds.IQ_abs.sel(qubit=q, qubit_state=0).values
        amp_e = ds.IQ_abs.sel(qubit=q, qubit_state=1).values
        amp_f = ds.IQ_abs.sel(qubit=q, qubit_state=2).values

        cg, kg, ag, og, ok_g = _fit_lorentzian_dip(dfs, amp_g)
        ce, ke, ae, oe, ok_e = _fit_lorentzian_dip(dfs, amp_e)
        cf, kf, af, of_, ok_f = _fit_lorentzian_dip(dfs, amp_f)

        f_g = rr_freq + cg
        f_e = rr_freq + ce
        f_f = rr_freq + cf

        success_g = ok_g and np.isfinite(cg) and abs(cg) < span_hz
        success_e = ok_e and np.isfinite(ce) and abs(ce) < span_hz
        success_f = ok_f and np.isfinite(cf) and abs(cf) < span_hz
        success = success_g and success_e and success_f

        chi_ge_hz = f_e - f_g if success else float("nan")
        chi_ef_hz = f_f - f_e if success else float("nan")
        # Optimal readout frequency: f_resonator(|e⟩) — maximises |e⟩ vs |g⟩ contrast
        f_optimal = f_e if success_e else rr_freq

        # Phase fit: φ(ω) = φ₀ − arctan(2·(ω − ωr) / κ)
        phi0_g = phi0_e = phi0_f = float("nan")
        kappa_ph_g = kappa_ph_e = kappa_ph_f = float("nan")
        center_ph_g = center_ph_e = center_ph_f = float("nan")
        if "phase" in ds.data_vars:
            ph_g = ds.phase.sel(qubit=q, qubit_state=0).values
            ph_e = ds.phase.sel(qubit=q, qubit_state=1).values
            ph_f = ds.phase.sel(qubit=q, qubit_state=2).values
            phi0_g, center_ph_g, kappa_ph_g, _ = _fit_phase(dfs, ph_g, cg, kg)
            phi0_e, center_ph_e, kappa_ph_e, _ = _fit_phase(dfs, ph_e, ce, ke)
            phi0_f, center_ph_f, kappa_ph_f, _ = _fit_phase(dfs, ph_f, cf, kf)

        fit_results[q] = FitParametersGEF(
            f_g=f_g,
            f_e=f_e,
            f_f=f_f,
            chi_ge_hz=chi_ge_hz,
            chi_ef_hz=chi_ef_hz,
            f_optimal=f_optimal,
            success=success,
            kappa_g_hz=kg,
            kappa_e_hz=ke,
            kappa_f_hz=kf,
            amplitude_g=ag,
            amplitude_e=ae,
            amplitude_f=af,
            offset_g=og,
            offset_e=oe,
            offset_f=of_,
            center_g_hz=cg,
            center_e_hz=ce,
            center_f_hz=cf,
            phi0_g=phi0_g,
            phi0_e=phi0_e,
            phi0_f=phi0_f,
            kappa_phase_g_hz=kappa_ph_g,
            kappa_phase_e_hz=kappa_ph_e,
            kappa_phase_f_hz=kappa_ph_f,
            center_phase_g_hz=center_ph_g,
            center_phase_e_hz=center_ph_e,
            center_phase_f_hz=center_ph_f,
        )

    return ds, fit_results
