"""Plotting utilities for the cavity f0g1 Rabi experiment."""
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis.models import oscillation_decay_exp


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fits: xr.Dataset,
    mode_name: str = "alice",
) -> Figure:
    """Plot Rabi fringe vs amplitude with fitted decaying-cosine overlay."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        ds_q = ds.sel(qubit=q_name)
        fit_q = fits.sel(qubit=q_name) if "fit" in fits else None
        _plot_single(ax, ds_q, fit_q)

    grid.fig.suptitle(f"f0g1 Rabi — {mode_name}")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax, ds, fit=None):
    x = ds.amp_factor.values

    if "state" in ds.data_vars:
        y = ds.state.values
        ylabel = "State population"
    elif "I" in ds.data_vars:
        y = ds.I.values * 1e3
        ylabel = "I (mV)"
    else:
        return

    ax.plot(x, y, ".", ms=4, color="C0", label="data")

    if fit is not None:
        a     = float(fit.fit.sel(fit_vals="a").values)
        f     = float(fit.fit.sel(fit_vals="f").values)
        phi   = float(fit.fit.sel(fit_vals="phi").values)
        offset = float(fit.fit.sel(fit_vals="offset").values)
        decay = float(fit.fit.sel(fit_vals="decay").values)

        if np.isfinite(f) and f > 0:
            x_dense = np.linspace(x.min(), x.max(), 500)
            y_fit = oscillation_decay_exp(x_dense, a, f, phi, offset, decay)
            ax.plot(x_dense, y_fit, "-", lw=1.5, color="C1", label="fit")

            pi_af = 0.5 / f
            ax.axvline(pi_af, color="C2", ls="--", lw=1, label=f"π: {pi_af:.3f}")

    ax.set_xlabel("Amplitude factor")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
