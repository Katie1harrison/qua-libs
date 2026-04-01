"""Analysis for displacement Ramsey calibration (node 30).

Experiment:
  For each displacement amplitude A, a Ramsey experiment is performed. The qubit
  is prepared in a superposition (x90), left to evolve for delay τ, then measured
  (x90). With n photons in the cavity the qubit accumulates an extra phase 2χ·n·τ.
  Averaging over the Poisson photon-number distribution gives:

      P_e(τ) = 0.5 * (1 + exp(-n̄·(1-cos(θ))) · cos(n̄·sin(θ)))

  where θ = 2π·chi_hz·τ and chi_hz is the dispersive-shift peak spacing in Hz.

Procedure:
  1. Estimate chi_hz from the FFT of the highest-amplitude trace (or use the value
     provided in node.parameters.chi_hz as an initial guess).
  2. Jointly fit chi_hz (shared) and n̄(A) for every non-zero amplitude A using
     scipy.optimize.least_squares.  Chi is not fixed — it is always fitted from
     the data.
  3. Fit n̄(A) = k·A² to obtain the calibration constant k.
  4. A₁ph = 1/√k is the amplitude_scale that deposits exactly 1 photon on average.

State updates (handled by node):
  - cavity_mode.cavity_mode_drive.operations["displacement"].amplitude = A₁ph * base_amp
  - CavityTransmonPair.displacement_k = k
  - CavityTransmonPair.chi = chi_hz_fitted  (if available)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit, least_squares

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Fit results for a single qubit's displacement Ramsey calibration."""
    k: float
    """Calibration constant: n̄ = k · A²."""
    amp_for_one_photon: float
    """Displacement amplitude_scale that deposits exactly 1 photon on average: 1/√k."""
    chi_hz: float
    """Dispersive shift (peak spacing) fitted from the data [Hz]."""
    nbar_vs_amp: List[Tuple[float, float]]
    """Raw calibration pairs [(A, n̄), ...] before the quadratic fit."""
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"[30] {q}: {status} | A₁ph = {res['amp_for_one_photon']:.4f} "
            f"| k = {res['k']:.3f} | chi_hz = {res['chi_hz'] * 1e-3:.3f} kHz"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


# ---------------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------------

def ramsey_photon_model(tau_ns: np.ndarray, nbar: float, chi_hz: float) -> np.ndarray:
    """Ramsey signal for a cavity in a coherent state with mean photon number n̄.

    P_e(τ) = 0.5 · (1 + exp(-n̄·(1-cos(θ))) · cos(n̄·sin(θ)))

    where θ = 2π·chi_hz·τ and chi_hz is the dispersive-shift peak spacing in Hz
    (= spacing between adjacent photon-number peaks in spectroscopy).

    Parameters
    ----------
    tau_ns : array-like
        Ramsey delays in nanoseconds.
    nbar : float
        Mean photon number of the coherent state.
    chi_hz : float
        Dispersive shift peak spacing [Hz].
    """
    theta = 2.0 * np.pi * chi_hz * np.asarray(tau_ns) * 1e-9
    return 0.5 * (1.0 + np.exp(-nbar * (1.0 - np.cos(theta))) * np.cos(nbar * np.sin(theta)))


def _nbar_model(A: np.ndarray, k: float) -> np.ndarray:
    return k * A ** 2


# ---------------------------------------------------------------------------
# Chi estimation helpers
# ---------------------------------------------------------------------------

def _estimate_chi_from_fft(tau_ns: np.ndarray, signal: np.ndarray) -> float:
    """Estimate chi_hz from the dominant oscillation frequency via FFT.

    Takes the signal with the largest contrast (oscillation amplitude) and
    returns the frequency of the strongest non-DC spectral component.
    """
    dt_s = float(np.mean(np.diff(tau_ns))) * 1e-9
    freqs = np.fft.rfftfreq(len(tau_ns), d=dt_s)
    power = np.abs(np.fft.rfft(signal - np.mean(signal)))
    power[0] = 0.0  # suppress DC
    peak_idx = int(np.argmax(power))
    return float(freqs[max(peak_idx, 1)])


def _chi_bounds_from_tau(tau_ns: np.ndarray) -> Tuple[float, float]:
    """Return (chi_lo, chi_hi) in Hz derived purely from the tau grid.

    chi_lo  : period = full tau span  (slowest resolvable oscillation)
    chi_hi  : period = 2 × Δτ        (Nyquist limit)
    """
    dt_s = float(np.mean(np.diff(tau_ns))) * 1e-9
    span_s = float(tau_ns[-1] - tau_ns[0]) * 1e-9
    chi_lo = max(1.0 / span_s, 10e3)   # at least 10 kHz
    chi_hi = 1.0 / (2.0 * dt_s)
    return chi_lo, chi_hi


# ---------------------------------------------------------------------------
# Joint fitting
# ---------------------------------------------------------------------------

def _joint_fit_chi_and_nbar(
    tau_ns: np.ndarray,
    signals: List[np.ndarray],
    chi0: float,
    nbar_max: float = 50.0,
) -> Tuple[float, List[float], bool]:
    """Jointly fit chi_hz (shared) and n̄ per amplitude using least_squares.

    Parameters
    ----------
    tau_ns : ndarray
        Ramsey delays [ns], shape (n_tau,).
    signals : list of ndarray
        One 1-D signal array per non-zero amplitude, each shape (n_tau,).
    chi0 : float
        Initial guess for chi_hz [Hz].
    nbar_max : float
        Upper bound on n̄.

    Returns
    -------
    chi_fit : float
    nbar_fits : list of float, one per element of ``signals``
    success : bool
    """
    n = len(signals)
    if n == 0:
        return chi0, [], False

    chi_lo, chi_hi = _chi_bounds_from_tau(tau_ns)
    # Clamp chi0 into the valid range
    chi0_clamped = float(np.clip(chi0, chi_lo, chi_hi))

    # Initial n̄ from per-trace signal contrast
    nbar0 = []
    for sig in signals:
        dev = float(np.mean(np.abs(sig - 0.5)))
        nbar0.append(max(dev * 2.0, 0.1))

    p0 = np.array([chi0_clamped] + nbar0, dtype=float)
    lo = np.array([chi_lo] + [0.0] * n)
    hi = np.array([chi_hi] + [nbar_max] * n)

    def residuals(p):
        chi = p[0]
        parts = []
        for i, sig in enumerate(signals):
            pred = ramsey_photon_model(tau_ns, float(p[1 + i]), chi)
            parts.append(pred - sig)
        return np.concatenate(parts)

    try:
        result = least_squares(
            residuals, p0,
            bounds=(lo, hi),
            max_nfev=200_000,
            method="trf",
            ftol=1e-9,
            xtol=1e-9,
        )
        # Mean squared residual per point as quality criterion
        msr = float(result.cost) / max(len(tau_ns) * n, 1)
        success = msr < 0.05  # empirical threshold
        return float(result.x[0]), [float(v) for v in result.x[1:]], success
    except Exception:
        return chi0_clamped, [float("nan")] * n, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _get_chi_initial_guess(node: QualibrationNode) -> Optional[float]:
    """Return chi_hz [Hz] from parameters or QUAM, to use as initial guess only."""
    if node.parameters.chi_hz is not None:
        val = float(node.parameters.chi_hz)
        if np.isfinite(val) and val > 0:
            return val

    mode_name = node.parameters.mode_name
    pairs = getattr(node.machine, "cavity_transmon_pairs", None)
    if pairs:
        for q in node.namespace.get("qubits", []):
            key = f"{q.name}_{mode_name}"
            pair = pairs.get(key)
            if pair is not None and getattr(pair, "chi", None) is not None:
                chi = float(pair.chi)
                if np.isfinite(chi) and chi > 0:
                    return chi

    cavity_mode = node.namespace.get("cavity_mode")
    if cavity_mode is not None:
        chi = getattr(cavity_mode, "chi", None)
        if chi is not None:
            chi = float(chi)
            if np.isfinite(chi) and chi > 0:
                return chi

    return None


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """Fit displacement Ramsey data, extracting chi_hz jointly from the data.

    For each qubit:
      1. Obtain an initial chi_hz guess (from parameters/QUAM, or FFT of the
         highest-amplitude trace if none is provided).
      2. Jointly fit chi_hz (shared across all amplitudes) and n̄ per amplitude
         using ``least_squares``.
      3. Fit n̄(A) = k·A² to extract the calibration constant k and A₁ph.
    """
    signal_name = "state" if node.parameters.use_state_discrimination else "I"
    fit_results: Dict[str, FitParameters] = {}
    chi_prior = _get_chi_initial_guess(node)

    for q in ds.qubit.values:
        ds_q = ds.sel(qubit=q)
        amps = ds_q.amp.values      # shape (n_amps,)
        tau_ns = ds_q.tau.values    # shape (n_tau,), in ns

        # ------------------------------------------------------------------
        # Step 1: initial chi guess
        # ------------------------------------------------------------------
        nonzero_idx = np.where(amps > 0)[0]
        if len(nonzero_idx) == 0:
            fit_results[str(q)] = FitParameters(
                k=float("nan"), amp_for_one_photon=float("nan"),
                chi_hz=float("nan"), nbar_vs_amp=[], success=False,
            )
            continue

        signals_nonzero = [
            ds_q.isel(amp=int(i))[signal_name].values.astype(float)
            for i in nonzero_idx
        ]

        if chi_prior is not None:
            chi0 = chi_prior
        else:
            # Pick the trace with the largest variance (clearest oscillations)
            variances = [float(np.var(s)) for s in signals_nonzero]
            best_idx = int(np.argmax(variances))
            chi0 = _estimate_chi_from_fft(tau_ns, signals_nonzero[best_idx])
            logging.getLogger(__name__).info(
                f"[30] {q}: no chi prior — FFT estimate = {chi0 * 1e-3:.1f} kHz"
            )

        # ------------------------------------------------------------------
        # Step 2: joint fit for chi (shared) and n̄ per amplitude
        # ------------------------------------------------------------------
        chi_fit, nbar_fits, joint_ok = _joint_fit_chi_and_nbar(
            tau_ns, signals_nonzero, chi0
        )

        if not joint_ok:
            logging.getLogger(__name__).warning(
                f"[30] {q}: joint fit did not converge — "
                f"results may be unreliable (chi={chi_fit * 1e-3:.1f} kHz)"
            )

        # Build (A, n̄) list including vacuum point
        nbar_list: List[Tuple[float, float]] = [(0.0, 0.0)]
        for A, nb in zip(amps[nonzero_idx], nbar_fits):
            if np.isfinite(nb):
                nbar_list.append((float(A), float(nb)))

        # ------------------------------------------------------------------
        # Step 3: fit n̄(A) = k·A²
        # ------------------------------------------------------------------
        try:
            valid = [(A, nb) for A, nb in nbar_list if A > 0 and np.isfinite(nb)]
            if len(valid) < 3:
                raise ValueError("too few valid (A, n̄) pairs")
            A_arr = np.array([v[0] for v in valid])
            nbar_arr = np.array([v[1] for v in valid])
            k0 = float(np.mean(nbar_arr / (A_arr ** 2 + 1e-12)))
            popt_k, _ = curve_fit(
                _nbar_model, A_arr, nbar_arr, p0=[max(k0, 1e-3)], maxfev=5000
            )
            k_fit = float(popt_k[0])
            amp_one = float(1.0 / np.sqrt(k_fit)) if k_fit > 0 else float("nan")
            success = bool(joint_ok and np.isfinite(amp_one) and k_fit > 0)
            fit_results[str(q)] = FitParameters(
                k=k_fit,
                amp_for_one_photon=amp_one,
                chi_hz=chi_fit,
                nbar_vs_amp=nbar_list,
                success=success,
            )
        except Exception:
            fit_results[str(q)] = FitParameters(
                k=float("nan"),
                amp_for_one_photon=float("nan"),
                chi_hz=chi_fit,
                nbar_vs_amp=nbar_list,
                success=False,
            )

    return ds, fit_results
