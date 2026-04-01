from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import lorentzian_peak
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, find_dip: bool = False):
    """
    Plots the EF qubit spectroscopy rotated I quadrature with fitted curves for the given qubits.

    The bottom x-axis shows the actual RF frequency swept in the EF experiment:
        RF = f_ge + detuning + anharmonicity
    stored in the dataset coordinate ``full_freq``.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data (must have ``full_freq`` coord).
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.
    find_dip : bool
        Must match the node parameter used during analysis.  When True the fit
        was performed on ``-I_rot``, so the baseline returned by ``peaks_dips``
        has the opposite sign to I_rot and must be negated before plotting.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), find_dip=find_dip)

    grid.fig.suptitle("Qubit EF spectroscopy (I_rot + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None, find_dip: bool = False):
    """
    Plots individual qubit EF data on a given axis with optional fit.

    The bottom axis shows the actual EF RF frequency (``full_freq``, which
    already accounts for the anharmonicity offset applied in the QUA program).
    The top axis shows the detuning relative to the scan centre.

    Parameters
    ----------
    find_dip : bool
        When True, negate the baseline so it is expressed in I_rot space
        (peaks_dips returns base_line in signal_for_fit = -I_rot space).
    """
    if fit is not None:
        # base_line from peaks_dips is in signal_for_fit space.
        # When find_dip=True, signal_for_fit = -I_rot, so the baseline has
        # the opposite sign to I_rot.  Negate it to match the plotted data.
        baseline_sign = -1.0 if find_dip else 1.0
        fitted_data = lorentzian_peak(
            ds.detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            baseline_sign * float(fit.base_line.mean().values),
        )
    else:
        fitted_data = None

    # Bottom axis: actual EF RF frequency (f_ge + detuning - anharmonicity)
    (fit.assign_coords(full_freq_GHz=fit.full_freq / u.GHz).I_rot / u.mV).plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("EF RF frequency [GHz]")
    ax.set_ylabel("I_rot [mV]")
    # Top axis: detuning relative to scan centre
    ax2 = ax.twiny()
    (fit.assign_coords(detuning_MHz=fit.detuning / u.MHz).I_rot / u.mV).plot(ax=ax2, x="detuning_MHz", label="")
    ax2.set_xlabel("Detuning from EF centre [MHz]")
    # Fitted curve
    if fitted_data is not None:
        ax2.plot(fit.detuning / u.MHz, fitted_data / u.mV, "r--")
