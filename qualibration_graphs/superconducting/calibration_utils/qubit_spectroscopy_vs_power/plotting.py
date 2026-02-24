from typing import List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


# ======================================================================
# Public API
# ======================================================================

def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    fits: xr.Dataset | None = None,
) -> Figure:
    """
    Plot for each qubit:
      • 2D spectroscopy vs power
      • 1D spectroscopy at selected power
    """

    ds = _ensure_iq_magnitude_mV(ds)

    # fit_raw_data may return a *new* Dataset object (when _mask_blacklisted_detunings
    # creates a copy), so ds_raw may be missing the fit-result variables.
    # Merge them in from fits so the overlay lines and the 1D power slice work correctly.
    if fits is not None:
        for var in ["selected_power", "rough_qubit_frequency"]:
            if var in fits and var not in ds:
                ds = ds.assign({var: fits[var]})

    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        _plot_combined_cell(ax, ds, qubit)

    grid.fig.suptitle("Qubit spectroscopy vs power (raw |IQ|, mV)")
    grid.fig.set_size_inches(15, 10)
    grid.fig.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.97, hspace=0.40)

    return grid.fig


# ======================================================================
# Combined cell (2D + 1D)
# ======================================================================

def _plot_combined_cell(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
):
    """
    Replace a single grid axis with two stacked sub-axes:
      - top: 2D spectroscopy vs power
      - bottom: 1D slice at selected power
    """

    fig = ax.figure
    gs = ax.get_subplotspec().subgridspec(
        2, 1,
        height_ratios=[3, 1],
        hspace=0.25,
    )

    ax_2d = fig.add_subplot(gs[0])
    ax_1d = fig.add_subplot(gs[1], sharex=ax_2d)

    ax.remove()

    plot_qubit_spectro_vs_power(ax_2d, ds, qubit)
    plot_qubit_spectro_at_selected_power(ax_1d, ds, qubit)

    plt.setp(ax_2d.get_xticklabels(), visible=False)


# ======================================================================
# 2D plot
# ======================================================================

def plot_qubit_spectro_vs_power(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
):
    ds_q = ds.loc[qubit]

    ds_q = ds_q.assign_coords(freq_GHz=ds_q.full_freq / u.GHz)

    ds_q.IQ_abs_mV.plot(
        ax=ax,
        x="freq_GHz",
        y="power",
        add_colorbar=False,
        robust=True,
    )

    ax.set_ylabel("Drive power [dBm]")

    # Top axis: detuning (via coordinate transform — avoids twiny+tight_layout stretch bug)
    if "detuning" in ds_q and ds_q.detuning.size > 0:
        det0 = float(ds_q.detuning[0])
        ff0 = float(ds_q.full_freq[0])
        rf_freq_GHz = (ff0 - det0) / u.GHz  # constant RF carrier in GHz
        ax_top = ax.secondary_xaxis(
            "top",
            functions=(
                lambda f: (f - rf_freq_GHz) * 1e3,   # GHz → MHz detuning
                lambda d: d / 1e3 + rf_freq_GHz,      # MHz detuning → GHz
            ),
        )
        ax_top.set_xlabel("Detuning [MHz]")

    # Selected power
    if "selected_power" in ds_q:
        p_sel = float(ds_q.selected_power)
        ax.axhline(p_sel, color="red", linewidth=2, zorder=10)

    # Selected frequency
    if "rough_qubit_frequency" in ds_q:
        f_sel = float(ds_q.rough_qubit_frequency) / u.GHz
        ax.axvline(
            f_sel,
            color="red",
            linestyle="--",
            linewidth=2,
            zorder=10,
        )



# ======================================================================
# 1D slice at selected power
# ======================================================================

def plot_qubit_spectro_at_selected_power(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
):
    ds_q = ds.loc[qubit]

    if "selected_power" in ds_q and np.isfinite(float(ds_q.selected_power)):
        p_sel = float(ds_q.selected_power)
        label = f"Slice at {p_sel:.1f} dBm"
    else:
        # No optimal power found — fall back to the highest power in the sweep
        p_sel = float(ds_q.power.max())
        label = f"Slice at max power: {p_sel:.1f} dBm"

    ds_slice = ds_q.sel(power=p_sel, method="nearest")

    x = ds_slice.full_freq / u.GHz

    ax.plot(x, ds_slice.IQ_abs_mV, linewidth=2)
    ax.set_ylabel("|IQ| [mV]")
    ax.set_xlabel("RF frequency [GHz]")
    ax.set_title(label, fontsize=9)

    if "rough_qubit_frequency" in ds_q:
        f_sel = float(ds_q.rough_qubit_frequency) / u.GHz
        ax.axvline(
            f_sel,
            color="limegreen",
            linestyle="--",
            linewidth=2,
        )


# ======================================================================
# Utilities
# ======================================================================

def _ensure_iq_magnitude_mV(ds: xr.Dataset) -> xr.Dataset:
    if "IQ_abs_mV" not in ds:
        ds["IQ_abs_mV"] = 1e3 * ds.IQ_abs
    return ds
