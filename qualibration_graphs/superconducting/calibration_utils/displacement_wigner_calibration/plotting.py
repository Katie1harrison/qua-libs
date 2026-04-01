"""Plotting utilities for the displacement Wigner calibration node (32)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict

import xarray as xr

from .analysis import FitParameters, wigner_gaussian


def plot_wigner_1d(
    ds: xr.Dataset,
    qubits,
    fit_results: Dict[str, FitParameters],
    parity_time_ns: int,
) -> plt.Figure:
    """Plot the 1D Wigner slice P_e(a) with Gaussian fit for each qubit.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with dims (qubit, amp).
    qubits : qubit list from node.namespace["qubits"]
    fit_results : dict
        Per-qubit FitParameters from fit_raw_data.
    parity_time_ns : int
        Parity time used in the experiment [ns], shown in the title.
    """
    qubit_names = list(ds.qubit.values)
    n_qubits = len(qubit_names)
    signal_name = "state" if "state" in ds else "I"

    fig, axes = plt.subplots(1, n_qubits, figsize=(6 * n_qubits, 5), squeeze=False)
    fig.suptitle(
        f"1D Wigner Slice — Displacement Wigner Calibration (32)\n"
        f"parity time = {parity_time_ns} ns",
        fontsize=13,
    )

    for ax, q_name in zip(axes[0], qubit_names):
        ds_q = ds.sel(qubit=q_name)
        a_arr = ds_q.amp.values.astype(float)
        signal = ds_q[signal_name].values.astype(float)

        ax.plot(a_arr, signal, "o", ms=4, label="data")

        res = fit_results.get(str(q_name))
        # fit_results values may be FitParameters dataclass or plain dict (after asdict())
        if isinstance(res, FitParameters):
            res = {"sigma": res.sigma, "amplitude": res.amplitude,
                   "offset": res.offset, "success": res.success}
        if res is not None and res.get("success"):
            sigma = res["sigma"]
            a_fine = np.linspace(a_arr[0], a_arr[-1], 500)
            fit_curve = wigner_gaussian(a_fine, res["amplitude"], sigma, res["offset"])
            ax.plot(a_fine, fit_curve, "-", lw=2, label=f"fit: σ = {sigma:.4f}")
            ax.axvline(sigma, color="green", linestyle="--", lw=1, alpha=0.7,
                       label=f"A₁ph = {sigma:.4f}")
            ax.axvline(-sigma, color="green", linestyle="--", lw=1, alpha=0.7)

        ax.set_xlabel("Displacement amplitude scale A", fontsize=11)
        ax.set_ylabel("State population", fontsize=11)
        ax.set_title(q_name, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
