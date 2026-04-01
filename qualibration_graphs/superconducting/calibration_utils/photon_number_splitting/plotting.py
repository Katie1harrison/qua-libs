"""Plotting for photon number splitting (node 29)."""
from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
    displacement_scale: float = 1.0,
) -> Figure:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        ds_q = ds.loc[qubit]
        x_hz = ds_q.detuning.values
        x_khz = x_hz * 1e-3

        signal_name = "state" if "state" in ds_q.data_vars else "I"
        if signal_name == "state":
            y = ds_q.state.values
            ylabel = "State population"
        else:
            y = ds_q.I.values * 1e3
            ylabel = "I (mV)"

        ax.plot(x_khz, y, ".", ms=3, color="C0", label="data")

        if fit_results and q_name in fit_results:
            res = fit_results[q_name]
            if res["success"]:
                peaks_khz = [p * 1e-3 for p in res["peak_positions_hz"] if np.isfinite(p)]
                for n_idx, pk in enumerate(peaks_khz):
                    ax.axvline(pk, color=f"C{n_idx + 1}", ls="--", lw=1.2, label=f"n={n_idx}")
                chi_khz = res["chi_hz"] * 1e-3
                ax.set_title(f"chi = {chi_khz:.3f} kHz", fontsize=9)
            else:
                ax.set_title("FAILED", fontsize=9, color="red")

        ax.set_xlabel("Qubit detuning (kHz)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    grid.fig.suptitle(
        f"Photon Number Splitting — {mode_name} (29)  "
        f"[displacement_scale={displacement_scale:.2f}]"
    )
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig
