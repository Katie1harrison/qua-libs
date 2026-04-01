"""Plotting utilities for the cavity mode chi (dispersive shift) measurement."""
from typing import Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from qualibration_libs.plotting import QubitGrid, grid_iter


def _lorentzian(x, a, x0, gamma, offset):
    return offset + a / (1 + ((x - x0) / gamma) ** 2)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
) -> Figure:
    """Plot dual-branch qubit spectroscopy for chi measurement.

    Two overlaid curves per qubit:
      - Blue: branch 0 (no photon in cavity, |g,0>)
      - Orange: branch 1 (one photon in cavity, |g,1>)
    Fitted Lorentzian peaks overlaid; chi annotated in title.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        q_fit_params = fit_results.get(q_name) if fit_results else None
        _plot_single(ax, ds.loc[qubit], q_fit_params)

    grid.fig.suptitle(f"Cavity Mode Chi — {mode_name}")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax, ds_q, fit_params=None):
    x_hz = ds_q.detuning.values  # in Hz
    x_khz = x_hz * 1e-3

    signal_name = "state" if "state" in ds_q.data_vars else "I"

    if signal_name == "state":
        y0 = ds_q.state.sel(photon_branch=0).values
        y1 = ds_q.state.sel(photon_branch=1).values
        ylabel = "State population"
        scale = 1.0
    elif signal_name == "I":
        y0 = ds_q.I.sel(photon_branch=0).values * 1e3
        y1 = ds_q.I.sel(photon_branch=1).values * 1e3
        ylabel = "I (mV)"
        scale = 1e3
    else:
        return

    ax.plot(x_khz, y0, ".", ms=4, color="C0", label="|g,0⟩")
    ax.plot(x_khz, y1, ".", ms=4, color="C1", label="|g,1⟩")

    # Overlay Lorentzian fits
    x_dense = np.linspace(x_hz.min(), x_hz.max(), 500)
    x_dense_khz = x_dense * 1e-3

    for branch_idx, (y_data, color) in enumerate([(y0, "C0"), (y1, "C1")]):
        try:
            idx_peak = np.argmax(y_data)
            a0 = float(y_data[idx_peak] - np.mean(y_data))
            x0_0 = float(x_hz[idx_peak])
            gamma0 = float((x_hz[-1] - x_hz[0]) / 10)
            offset0 = float(np.mean(y_data))
            popt, _ = curve_fit(
                _lorentzian, x_hz, y_data,
                p0=[a0, x0_0, gamma0, offset0],
                maxfev=10000,
            )
            y_fit = _lorentzian(x_dense, *popt)
            ax.plot(x_dense_khz, y_fit, "-", lw=1.5, color=color, alpha=0.8)
        except Exception:
            pass

    if fit_params is not None:
        chi_hz = fit_params.get("chi_hz", float("nan"))
        f0_hz = fit_params.get("f_ge_0_hz", float("nan"))
        f1_hz = fit_params.get("f_ge_1_hz", float("nan"))
        success = fit_params.get("success", False)
        status = "SUCCESS" if success else "FAILED"

        if np.isfinite(chi_hz):
            chi_khz = chi_hz * 1e-3
            ax.set_title(f"chi = {chi_khz:.3f} kHz  [{status}]", fontsize=9,
                         color="green" if success else "red")
            # Mark the two peak positions
            if np.isfinite(f0_hz):
                ax.axvline(f0_hz * 1e-3, color="C0", ls="--", lw=1,
                           label=f"f_ge,0={f0_hz*1e-3:.2f} kHz")
            if np.isfinite(f1_hz):
                ax.axvline(f1_hz * 1e-3, color="C1", ls="--", lw=1,
                           label=f"f_ge,1={f1_hz*1e-3:.2f} kHz")
        else:
            ax.set_title(f"[{status}]", fontsize=9, color="red")

    ax.set_xlabel("Detuning (kHz)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
