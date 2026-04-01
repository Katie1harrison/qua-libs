"""Plotting for the selective power Rabi node (14b)."""
from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import oscillation


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    ds_fit: xr.Dataset,
    fit_results: Optional[Dict] = None,
) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        ds_q = ds.loc[qubit]
        fit_q = ds_fit.loc[qubit]

        amp_mv = ds_q.full_amp.values * 1e3
        amp_ax = ds_q.amp_prefactor.values

        if "state" in ds_q.data_vars:
            y = ds_q.state.values
            ylabel = "State population"
        else:
            y = ds_q.I.values * 1e3
            ylabel = "I (mV)"

        ax.plot(amp_mv, y, ".", ms=3, color="C0", label="data")

        # Fitted curve
        try:
            a = float(fit_q.fit.sel(fit_vals="a").item())
            f = float(fit_q.fit.sel(fit_vals="f").item())
            phi = float(fit_q.fit.sel(fit_vals="phi").item())
            off = float(fit_q.fit.sel(fit_vals="offset").item())
            y_fit = oscillation(amp_ax, a, f, phi, off)
            scale = 1.0 if "state" in ds_q.data_vars else 1e3
            ax.plot(amp_mv, y_fit * scale, "-", lw=1.5, color="C1", label="fit")
        except Exception:
            pass

        # Mark pi amplitude
        if fit_results and q_name in fit_results:
            res = fit_results[q_name]
            if res["success"] and np.isfinite(res["pi_amp"]):
                ax.axvline(
                    res["pi_amp"] * 1e3, color="green", ls="--", lw=1.5,
                    label=f"pi = {res['pi_amp']*1e3:.2f} mV",
                )
                ax.set_title(f"pi_amp = {res['pi_amp']*1e3:.2f} mV", fontsize=9)
            else:
                ax.set_title("FAILED", fontsize=9, color="red")

        ax.set_xlabel("Amplitude (mV)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    grid.fig.suptitle("Selective Power Rabi (14b)")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig
