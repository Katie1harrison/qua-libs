"""Plotting utilities for cavity f0g1 spectroscopy."""
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fits: xr.Dataset,
    mode_name: str = "alice",
) -> Figure:
    """Plot cavity f0g1 spectroscopy data with Lorentzian dip overlay."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        _plot_single(ax, ds.sel(qubit=q_name), fits.sel(qubit=q_name))

    grid.fig.suptitle(f"f0g1 spectroscopy — {mode_name} (state dip + fit)")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax: Axes, ds: xr.Dataset, fit: xr.Dataset):
    """Plot one qubit panel."""
    if "full_freq" in ds.coords:
        x = ds.full_freq.values * 1e-9  # → GHz
        xlabel = "RF frequency (GHz)"
    else:
        x = ds.detuning.values * 1e-6  # → MHz
        xlabel = "Detuning (MHz)"

    if "state" in ds.data_vars:
        y = ds.state.values
        ylabel = "State population"
    elif "I_rot" in ds.data_vars:
        y = ds.I_rot.values * 1e3
        ylabel = "I_rot (mV)"
    else:
        return

    ax.plot(x, y, ".", ms=4, color="C0", label="data")

    pos = float(fit.position.values) if "position" in fit else np.nan
    if np.isfinite(pos):
        if "full_freq" in ds.coords:
            f0g1_rf = float(ds.full_freq.isel(detuning=0).values) - float(ds.detuning.isel(detuning=0).values)
            xpos = (f0g1_rf + pos) * 1e-9
        else:
            xpos = pos * 1e-6
        ax.axvline(xpos, color="C1", lw=1.5, ls="--", label=f"fit: {xpos:.4f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
