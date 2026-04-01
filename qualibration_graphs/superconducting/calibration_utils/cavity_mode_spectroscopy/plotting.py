"""Plotting utilities for the cavity mode spectroscopy measurement (node 27)."""
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def _lorentzian_dip(x, amplitude, center, hwhm, offset):
    return offset - amplitude * hwhm ** 2 / (hwhm ** 2 + (x - center) ** 2)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results=None,
    mode_name: str = "alice",
) -> Figure:
    """Plot cavity mode spectroscopy: qubit excitation vs cavity drive frequency.

    A dip in excitation probability marks the cavity resonance frequency.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        fp = fit_results.get(q_name) if fit_results else None
        _plot_single(ax, ds.loc[qubit], fp)

    grid.fig.suptitle(f"Cavity Mode Spectroscopy — {mode_name}")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax, ds_q, fit_params=None):
    use_full_freq = "full_freq" in ds_q.coords
    if use_full_freq:
        x_hz = ds_q.full_freq.values
        x_plot = x_hz * 1e-9
        xlabel = "RF frequency (GHz)"
    else:
        x_hz = ds_q.detuning.values.astype(float)
        x_plot = x_hz * 1e-6
        xlabel = "Detuning (MHz)"

    signal_name = "state" if "state" in ds_q.data_vars else "I"
    if signal_name == "state":
        y = ds_q.state.values
        ylabel = "State population"
    else:
        y = ds_q.I.values * 1e3
        ylabel = "I (mV)"

    ax.plot(x_plot, y, ".", ms=4, color="C0", alpha=0.8, label="data")

    if fit_params is not None:
        success = fit_params.get("success", False)
        freq_hz = fit_params.get("frequency_hz", float("nan"))
        fwhm_hz = fit_params.get("fwhm_hz", float("nan"))

        if success and np.isfinite(freq_hz):
            # Vertical marker at cavity resonance only (no Lorentzian overlay)
            freq_plot = freq_hz * 1e-9 if use_full_freq else fit_params.get("center_detuning_hz", float("nan")) * 1e-6
            ax.axvline(freq_plot, color="r", ls="--", lw=1.5,
                       label=f"f_cav={freq_hz*1e-9:.4f} GHz")
            fwhm_mhz = fwhm_hz * 1e-6
            ax.set_title(
                f"f_cav = {freq_hz*1e-9:.5f} GHz\nFWHM = {fwhm_mhz:.2f} MHz  [SUCCESS]",
                fontsize=9, color="green"
            )
        else:
            ax.set_title("[FAILED]", fontsize=9, color="red")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
