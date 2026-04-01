from typing import List
import xarray as xr
from matplotlib.axes import Axes
from qualibration_libs.analysis import decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """Plot depletion measurement data with exponential fit for all qubits."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        _plot_individual(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Readout depletion measurement")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def _plot_individual(ax: Axes, ds: xr.Dataset, qubit: dict, fit: xr.Dataset):
    q = qubit["qubit"]
    ds_q = ds.sel(qubit=q)

    if hasattr(ds_q, "state"):
        data = ds_q.state
        ylabel = "Ramsey state population"
    else:
        data = ds_q.I * 1e3
        ylabel = "Trans. amp. I [mV]"

    data.plot(ax=ax, label="Data")

    fitted = decay_exp(
        ds_q.idle_time,
        fit.fit_data.sel(fit_vals="a"),
        fit.fit_data.sel(fit_vals="offset"),
        fit.fit_data.sel(fit_vals="decay"),
    )
    ax.plot(ds_q.idle_time.values, fitted if not hasattr(ds_q, "state") else fitted, "r--", label="Fit")

    tau = float(fit.tau.values)
    # Mark the depletion time
    ax.axvline(tau, color="orange", linestyle=":", label=f"T_dep = {tau:.0f} ns")

    ax.set_xlabel("Wait time after readout [ns]")
    ax.set_ylabel(ylabel)
    ax.set_title(q)
    ax.legend(fontsize=8)

    success = bool(fit.success.values)
    tau_err = float(fit.tau_error.values)
    ax.text(
        0.05,
        0.95,
        f"T_dep = {tau:.0f} ± {tau_err:.0f} ns\n{'SUCCESS' if success else 'FAIL'}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6),
    )
