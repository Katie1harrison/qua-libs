"""Plotting utilities for the cavity coherent T1 node (33)."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

import xarray as xr

from .analysis import CoherentT1Fit, coherent_T1_model


def plot_coherent_T1(
    ds: xr.Dataset,
    fit_results: Dict,
    mode_name: str = "alice",
) -> plt.Figure:
    """Plot the coherent T1 double-exponential decay for each qubit.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset with dim (qubit, idle_time).
    fit_results : dict
        Per-qubit fit results — either CoherentT1Fit dataclasses or plain dicts
        (after asdict()).
    mode_name : str
        Cavity mode name (shown in title).
    """
    qubit_names = list(ds.qubit.values)
    n_qubits = len(qubit_names)
    signal_name = "state" if "state" in ds else "I"

    fig, axes = plt.subplots(1, n_qubits, figsize=(7 * n_qubits, 5), squeeze=False)
    fig.suptitle(
        f"Cavity Coherent T1 — mode '{mode_name}' (node 33)",
        fontsize=13,
    )

    for ax, q_name in zip(axes[0], qubit_names):
        ds_q = ds.sel(qubit=q_name)
        t_ns = ds_q.idle_time.values.astype(float)
        signal = ds_q[signal_name].values.astype(float)

        ax.plot(t_ns * 1e-3, signal, "o", ms=4, color="steelblue", label="data")

        res = fit_results.get(str(q_name))
        # Normalise to plain dict (may arrive as dataclass or dict)
        if isinstance(res, CoherentT1Fit):
            res = {
                "T1_ns": res.T1_ns,
                "T1_error_ns": res.T1_error_ns,
                "nbar0": res.nbar0,
                "amplitude": res.amplitude,
                "offset": res.offset,
                "success": res.success,
            }

        if res is not None and res.get("success"):
            T1_us = res["T1_ns"] * 1e-3
            T1_err_us = res["T1_error_ns"] * 1e-3
            t_fine = np.linspace(t_ns[0], t_ns[-1], 500)
            fit_curve = coherent_T1_model(
                t_fine,
                res["amplitude"],
                res["nbar0"],
                res["T1_ns"],
                res["offset"],
            )
            ax.plot(
                t_fine * 1e-3,
                fit_curve,
                "-",
                lw=2,
                color="tomato",
                label=f"fit: T₁ = {T1_us:.1f} ± {T1_err_us:.1f} µs\nnbar0 = {res['nbar0']:.2f}",
            )
            ax.axvline(T1_us, color="gray", linestyle="--", lw=1, alpha=0.6,
                       label=f"T₁ = {T1_us:.1f} µs")
        elif res is not None:
            ax.text(
                0.5, 0.5, "Fit failed",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=12, color="red",
            )

        ax.set_xlabel("Wait time (µs)", fontsize=11)
        ax.set_ylabel("State population", fontsize=11)
        ax.set_title(q_name, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
