from typing import Dict, List, Optional

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import oscillation
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset,
    fit_results: Optional[Dict] = None,
) -> Figure:
    """
    Plot EF time Rabi oscillations (I quadrature or state) vs pulse duration for each qubit.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset from process_raw_dataset (must contain duration_ns coordinate).
    qubits : list of AnyTransmon
        Qubits to plot.
    fits : xr.Dataset
        Fit dataset from fit_raw_data (contains 'fit' variable with fit_vals dimension).
    fit_results : dict, optional
        Per-qubit FitParameters dicts (as returned by asdict). When provided, each
        subplot title shows the chi2 value and SUCCESS / FAILED status.

    Returns
    -------
    Figure
        Matplotlib figure with one subplot per qubit.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        q_fit_params = fit_results.get(q_name) if fit_results else None
        _plot_individual(ax, ds, qubit, fits.sel(qubit=q_name), q_fit_params)

    grid.fig.suptitle("EF Time Rabi")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_individual(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict,
    fit: xr.Dataset = None,
    fit_params: Optional[Dict] = None,
):
    """Plot I [mV] or state vs duration [ns] with fitted sinusoid overlay."""
    ds_q = ds.loc[qubit]

    if "duration_ns" not in ds_q.coords:
        x = ds_q.duration_cc.values * 4
    else:
        x = ds_q.duration_ns.values

    if hasattr(ds_q, "state"):
        y = ds_q.state.values
        y_label = "Qubit state"
        scale = 1.0
    else:
        y = ds_q.I.values * 1e3
        y_label = "I quadrature [mV]"
        scale = 1e3

    ax.plot(x, y, ".", markersize=4, label="data")

    if fit is not None:
        a = float(fit.fit.sel(fit_vals="a").item())
        f = float(fit.fit.sel(fit_vals="f").item())
        phi = float(fit.fit.sel(fit_vals="phi").item())
        offset = float(fit.fit.sel(fit_vals="offset").item())

        if np.isfinite(a) and np.isfinite(f):
            x_fit = ds_q.duration_cc.values
            fitted = oscillation(x_fit, a, f, phi, offset) * scale
            ax.plot(x, fitted, "r-", linewidth=1.5, label="fit")

            if abs(f) > 0:
                pi_cc = 1.0 / (2.0 * abs(f))
                pi_ns = round(pi_cc) * 4
                ax.axvline(pi_ns, color="green", linestyle="--", linewidth=1.5,
                           label=f"π = {pi_ns:.0f} ns")

    if fit_params is not None:
        chi2 = fit_params.get("chi2", float("nan"))
        success = fit_params.get("success", False)
        num_periods = fit_params.get("num_periods", float("nan"))
        status_str = "SUCCESS" if success else "FAILED"
        chi2_str = f"{chi2:.2f}" if np.isfinite(chi2) else "nan"
        periods_str = f"{num_periods:.2f}" if np.isfinite(num_periods) else "nan"
        color = "green" if success else "red"
        ax.set_title(
            f"chi2={chi2_str}  periods={periods_str}  [{status_str}]",
            fontsize=8,
            color=color,
        )

    ax.set_xlabel("EF pulse duration [ns]")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)
