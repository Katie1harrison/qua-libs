import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import curve_fit


@dataclass
class FitParameters:
    """Stores the GEF IQ blob fit and classifier parameters for a single qubit."""

    I_g_center: float
    Q_g_center: float
    I_e_center: float
    Q_e_center: float
    I_f_center: float
    Q_f_center: float
    # 1D rotated threshold classifier
    rotation_angle_rad: float = 0.0
    threshold_low: float = 0.0
    threshold_high: float = 0.0
    threshold_sorted_order: List[int] = field(default_factory=lambda: [0, 1, 2])
    confusion_matrix_1d: List[List[float]] = field(default_factory=lambda: [[0.0] * 3] * 3)
    # Nearest-centroid (2D) classifier
    confusion_matrix_nc: List[List[float]] = field(default_factory=lambda: [[0.0] * 3] * 3)
    # LDA (2D) classifier
    lda_sigma: List[List[float]] = field(default_factory=lambda: [[1.0, 0.0], [0.0, 1.0]])
    lda_priors: List[float] = field(default_factory=lambda: [1 / 3, 1 / 3, 1 / 3])
    confusion_matrix_lda: List[List[float]] = field(default_factory=lambda: [[0.0] * 3] * 3)
    # Blob separation metrics
    d_ge_V: float = 0.0
    d_gf_V: float = 0.0
    d_ef_V: float = 0.0
    sigma_rms_V: float = 0.0
    d_ge_over_sigma: Optional[float] = None
    d_gf_over_sigma: Optional[float] = None
    d_ef_over_sigma: Optional[float] = None
    success: bool = True


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log GEF center positions and classifier diagnostics for all qubits."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, fp in fit_results.items():
        centers_str = (
            f"g:({fp.I_g_center * 1e3:.1f},{fp.Q_g_center * 1e3:.1f}) mV | "
            f"e:({fp.I_e_center * 1e3:.1f},{fp.Q_e_center * 1e3:.1f}) mV | "
            f"f:({fp.I_f_center * 1e3:.1f},{fp.Q_f_center * 1e3:.1f}) mV"
        )
        sep_str = (
            f"d_ge/σ={fp.d_ge_over_sigma:.2f}, "
            f"d_gf/σ={fp.d_gf_over_sigma:.2f}, "
            f"d_ef/σ={fp.d_ef_over_sigma:.2f}"
        ) if fp.d_ge_over_sigma is not None else ""
        status = "SUCCESS" if fp.success else "FAIL"
        log_callable(f"GEF blobs for {q}: {status} | {centers_str}" + (f" | {sep_str}" if sep_str else ""))
        # Log LDA diagonal (fidelities)
        cm = np.array(fp.confusion_matrix_lda)
        log_callable(
            f"  LDA fidelity: P(g|g)={cm[0,0]:.3f}, P(e|e)={cm[1,1]:.3f}, P(f|f)={cm[2,2]:.3f}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stack_iq(I, Q):
    return np.column_stack([np.asarray(I, dtype=float), np.asarray(Q, dtype=float)])


def _confusion_matrix_3state(true_labels, pred_labels):
    """3×3 confusion matrix, rows=prepared state (0=g,1=e,2=f), cols=measured."""
    M = np.zeros((3, 3), dtype=float)
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    for i in range(3):
        idx = np.where(true_labels == i)[0]
        if len(idx) == 0:
            continue
        for j in range(3):
            M[i, j] = np.mean(pred_labels[idx] == j)
    return M


def _compute_classifiers(Ig, Qg, Ie, Qe, If, Qf):
    """
    Compute 1D-threshold, nearest-centroid, and LDA classifiers from GEF IQ data (in volts).

    Returns a dict with all classifier parameters and confusion matrices.
    """
    Ig, Qg = np.asarray(Ig, dtype=float), np.asarray(Qg, dtype=float)
    Ie, Qe = np.asarray(Ie, dtype=float), np.asarray(Qe, dtype=float)
    If, Qf = np.asarray(If, dtype=float), np.asarray(Qf, dtype=float)

    Xg = _stack_iq(Ig, Qg)
    Xe = _stack_iq(Ie, Qe)
    Xf = _stack_iq(If, Qf)
    X_all = np.vstack([Xg, Xe, Xf])
    n_total = len(X_all)
    true_all = np.concatenate([
        np.zeros(len(Xg), dtype=int),
        np.ones(len(Xe), dtype=int),
        2 * np.ones(len(Xf), dtype=int),
    ])

    mu_g = np.mean(Xg, axis=0)
    mu_e = np.mean(Xe, axis=0)
    mu_f = np.mean(Xf, axis=0)
    mus = np.stack([mu_g, mu_e, mu_f], axis=0)  # shape (3, 2)

    # ------------------------------------------------------------------
    # 1. Nearest-centroid classifier
    # ------------------------------------------------------------------
    d2 = np.sum((X_all[:, None, :] - mus[None, :, :]) ** 2, axis=2)
    pred_nc = np.argmin(d2, axis=1)
    confusion_nc = _confusion_matrix_3state(true_all, pred_nc)

    # ------------------------------------------------------------------
    # 2. LDA classifier (pooled covariance)
    # ------------------------------------------------------------------
    def _cov(X):
        n = len(X)
        if n <= 1:
            return np.eye(2) * 1e-12
        Xc = X - np.mean(X, axis=0, keepdims=True)
        return (Xc.T @ Xc) / (n - 1)

    Sigma = (
        (len(Xg) - 1) * _cov(Xg)
        + (len(Xe) - 1) * _cov(Xe)
        + (len(Xf) - 1) * _cov(Xf)
    ) / max((len(Xg) - 1) + (len(Xe) - 1) + (len(Xf) - 1), 1)
    Sigma += 1e-9 * np.eye(2)
    Sigma_inv = np.linalg.inv(Sigma)

    priors = np.array([len(Xg) / n_total, len(Xe) / n_total, len(Xf) / n_total])

    scores = []
    for k in range(3):
        mu = mus[k]
        score = X_all @ (Sigma_inv @ mu) - 0.5 * (mu @ Sigma_inv @ mu) + np.log(priors[k])
        scores.append(score)
    pred_lda = np.argmax(np.column_stack(scores), axis=1)
    confusion_lda = _confusion_matrix_3state(true_all, pred_lda)

    # ------------------------------------------------------------------
    # 3. 1D rotated-threshold classifier
    # ------------------------------------------------------------------
    theta = float(np.angle((mu_e[0] - mu_g[0]) + 1j * (mu_e[1] - mu_g[1])))

    def _rotate_I(I, Q, angle):
        z = np.asarray(I, dtype=float) + 1j * np.asarray(Q, dtype=float)
        return np.real(z * np.exp(-1j * angle))

    Ig_r = _rotate_I(Ig, Qg, theta)
    Ie_r = _rotate_I(Ie, Qe, theta)
    If_r = _rotate_I(If, Qf, theta)

    centroids_1d = np.array([float(np.mean(Ig_r)), float(np.mean(Ie_r)), float(np.mean(If_r))])
    order = np.argsort(centroids_1d)  # indices of states sorted along rotated I axis
    thr_low = float(0.5 * (centroids_1d[order[0]] + centroids_1d[order[1]]))
    thr_high = float(0.5 * (centroids_1d[order[1]] + centroids_1d[order[2]]))

    all_rot_I = np.concatenate([Ig_r, Ie_r, If_r])
    pred_sorted = np.zeros(len(all_rot_I), dtype=int)
    pred_sorted[all_rot_I > thr_low] = 1
    pred_sorted[all_rot_I > thr_high] = 2
    pred_1d = np.array([int(order[v]) for v in pred_sorted])
    confusion_1d = _confusion_matrix_3state(true_all, pred_1d)

    # ------------------------------------------------------------------
    # 4. Blob separation metrics
    # ------------------------------------------------------------------
    sigma_rms = float(np.sqrt(np.trace(Sigma) / 2.0))
    d_ge = float(np.linalg.norm(mu_e - mu_g))
    d_gf = float(np.linalg.norm(mu_f - mu_g))
    d_ef = float(np.linalg.norm(mu_f - mu_e))

    return {
        "mu_g": mu_g,
        "mu_e": mu_e,
        "mu_f": mu_f,
        # Nearest centroid
        "confusion_matrix_nc": confusion_nc,
        # LDA
        "lda_sigma": Sigma,
        "lda_priors": priors,
        "confusion_matrix_lda": confusion_lda,
        # 1D threshold
        "rotation_angle_rad": theta,
        "threshold_low": thr_low,
        "threshold_high": thr_high,
        "threshold_sorted_order": order.tolist(),
        "confusion_matrix_1d": confusion_1d,
        # Blob separation
        "d_ge_V": d_ge,
        "d_gf_V": d_gf,
        "d_ef_V": d_ef,
        "sigma_rms_V": sigma_rms,
        "d_ge_over_sigma": float(d_ge / sigma_rms) if sigma_rms > 0 else None,
        "d_gf_over_sigma": float(d_gf / sigma_rms) if sigma_rms > 0 else None,
        "d_ef_over_sigma": float(d_ef / sigma_rms) if sigma_rms > 0 else None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_biggest_gaussian(da):
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    hist, bin_edges = np.histogram(da, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    initial_guess = [(hist.max(), bin_centers[hist.argmax()], (bin_centers[-1] - bin_centers[0]) / 4)]
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)
    return popt[1]  # mu


def fit_gaussian_centers(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Fit Gaussian centers and compute all three classifiers for each qubit."""
    fit_results = {}
    for q in ds.qubit.values:
        ds_q = ds.sel(qubit=q)
        I_g_cent = find_biggest_gaussian(ds_q.Ig)
        Q_g_cent = find_biggest_gaussian(ds_q.Qg)
        I_e_cent = find_biggest_gaussian(ds_q.Ie)
        Q_e_cent = find_biggest_gaussian(ds_q.Qe)
        I_f_cent = find_biggest_gaussian(ds_q.If)
        Q_f_cent = find_biggest_gaussian(ds_q.Qf)

        clf = _compute_classifiers(
            ds_q.Ig.values, ds_q.Qg.values,
            ds_q.Ie.values, ds_q.Qe.values,
            ds_q.If.values, ds_q.Qf.values,
        )

        fit_results[q] = {
            "I_g_center": I_g_cent,
            "Q_g_center": Q_g_cent,
            "I_e_center": I_e_cent,
            "Q_e_center": Q_e_cent,
            "I_f_center": I_f_cent,
            "Q_f_center": Q_f_cent,
            **clf,
        }

    # Build scalar vars per qubit
    qubit_names = list(fit_results.keys())
    scalar_params = [
        "I_g_center", "Q_g_center", "I_e_center", "Q_e_center", "I_f_center", "Q_f_center",
        "rotation_angle_rad", "threshold_low", "threshold_high",
        "d_ge_V", "d_gf_V", "d_ef_V", "sigma_rms_V",
        "d_ge_over_sigma", "d_gf_over_sigma", "d_ef_over_sigma",
    ]
    data_vars = {p: (["qubit"], [fit_results[q][p] for q in qubit_names]) for p in scalar_params}

    # Matrix vars: shape (qubit, 3, 3) for confusion matrices, (qubit, 2, 2) for LDA sigma
    data_vars["confusion_matrix_nc"] = (
        ["qubit", "prepared", "measured"],
        np.stack([fit_results[q]["confusion_matrix_nc"] for q in qubit_names], axis=0),
    )
    data_vars["confusion_matrix_lda"] = (
        ["qubit", "prepared", "measured"],
        np.stack([fit_results[q]["confusion_matrix_lda"] for q in qubit_names], axis=0),
    )
    data_vars["confusion_matrix_1d"] = (
        ["qubit", "prepared", "measured"],
        np.stack([fit_results[q]["confusion_matrix_1d"] for q in qubit_names], axis=0),
    )
    data_vars["lda_sigma"] = (
        ["qubit", "IQ_row", "IQ_col"],
        np.stack([fit_results[q]["lda_sigma"] for q in qubit_names], axis=0),
    )
    data_vars["lda_priors"] = (
        ["qubit", "state"],
        np.stack([fit_results[q]["lda_priors"] for q in qubit_names], axis=0),
    )
    data_vars["threshold_sorted_order"] = (
        ["qubit", "rank"],
        np.array([[fit_results[q]["threshold_sorted_order"] for q in qubit_names]]).squeeze(0),
    )

    fit_ds = xr.Dataset(data_vars, coords={"qubit": qubit_names})
    fit_ds = xr.merge([ds, fit_ds])
    return fit_ds


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    def extract_value(element):
        if isinstance(element, tuple):
            return element[0]
        return element

    ds = xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe", "If", "Qf"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict]:
    """Fit Gaussian centers and compute classifiers for each qubit."""
    ds_fit = fit_gaussian_centers(ds, node)
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Build FitParameters objects and add center_matrix to ds_fit."""
    fit_results = {}
    for q in fit.qubit.values:
        fq = fit.sel(qubit=q)
        fit_results[q] = FitParameters(
            I_g_center=float(fq.I_g_center),
            Q_g_center=float(fq.Q_g_center),
            I_e_center=float(fq.I_e_center),
            Q_e_center=float(fq.Q_e_center),
            I_f_center=float(fq.I_f_center),
            Q_f_center=float(fq.Q_f_center),
            rotation_angle_rad=float(fq.rotation_angle_rad),
            threshold_low=float(fq.threshold_low),
            threshold_high=float(fq.threshold_high),
            threshold_sorted_order=fq.threshold_sorted_order.values.tolist(),
            confusion_matrix_1d=fq.confusion_matrix_1d.values.tolist(),
            confusion_matrix_nc=fq.confusion_matrix_nc.values.tolist(),
            lda_sigma=fq.lda_sigma.values.tolist(),
            lda_priors=fq.lda_priors.values.tolist(),
            confusion_matrix_lda=fq.confusion_matrix_lda.values.tolist(),
            d_ge_V=float(fq.d_ge_V),
            d_gf_V=float(fq.d_gf_V),
            d_ef_V=float(fq.d_ef_V),
            sigma_rms_V=float(fq.sigma_rms_V),
            d_ge_over_sigma=float(fq.d_ge_over_sigma) if fq.d_ge_over_sigma.item() is not None else None,
            d_gf_over_sigma=float(fq.d_gf_over_sigma) if fq.d_gf_over_sigma.item() is not None else None,
            d_ef_over_sigma=float(fq.d_ef_over_sigma) if fq.d_ef_over_sigma.item() is not None else None,
            success=True,
        )

    fit = fit.groupby("qubit").apply(center_matrix)
    return fit, fit_results


def center_matrix(da: xr.DataArray) -> xr.DataArray:
    center = np.array(
        [
            [da.I_g_center.item(), da.Q_g_center.item()],
            [da.I_e_center.item(), da.Q_e_center.item()],
            [da.I_f_center.item(), da.Q_f_center.item()],
        ]
    )
    return da.assign(center_matrix=(["state_3", "IQ"], center))
