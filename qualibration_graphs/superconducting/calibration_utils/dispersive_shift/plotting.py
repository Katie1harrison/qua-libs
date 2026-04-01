"""Plotting utilities for the dispersive shift measurement."""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.analysis.models import lorentzian_dip


def _phase_model(x, phi0, center, kappa):
    """φ(ω) = φ₀ − arctan(2·(ω − ωr) / κ)"""
    return phi0 - np.arctan(2.0 * (x - center) / kappa)


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fits,
) -> Figure:
    """Plot amplitude and phase spectra for |g⟩ and |e⟩ with Lorentzian / arctan fits."""
    n_qubits = len(list(qubits))
    fig, axes = plt.subplots(2, n_qubits, figsize=(6 * n_qubits, 8), squeeze=False)

    for col, q_obj in enumerate(qubits):
        q_name = q_obj.name
        fp = fits.get(q_name) if isinstance(fits, dict) else None
        ds_q = ds.sel(qubit=q_name)
        _plot_amplitude(axes[0, col], ds_q, fp)
        _plot_phase(axes[1, col], ds_q, fp)
        axes[0, col].set_title(f"{q_name}" + (
            f"\nchi={fp.chi_hz*1e-3:.1f} kHz  "
            f"κ_g={fp.kappa_g_hz*2e-3:.1f} kHz  "
            f"κ_e={fp.kappa_e_hz*2e-3:.1f} kHz"
            if fp is not None and fp.success else ""
        ))

    fig.suptitle("Dispersive shift — |g⟩ vs |e⟩ resonator spectra")
    fig.tight_layout()
    return fig


def _x_axis(ds_q):
    """Return (x_plot, x_hz, xlabel)."""
    if "full_freq" in ds_q.coords:
        x_hz = ds_q.full_freq.values
        return x_hz * 1e-9, x_hz, "RF frequency (GHz)"
    else:
        x_hz = ds_q.detuning.values
        return x_hz * 1e-6, x_hz, "Detuning (MHz)"


def _plot_amplitude(ax, ds_q, fp):
    x_plot, x_hz, xlabel = _x_axis(ds_q)
    use_full_freq = "full_freq" in ds_q.coords

    if "qubit_state" in ds_q.dims:
        amp_g = ds_q.IQ_abs.sel(qubit_state=0).values * 1e3
        amp_e = ds_q.IQ_abs.sel(qubit_state=1).values * 1e3
    elif "IQ_abs_g" in ds_q.data_vars:
        amp_g = ds_q.IQ_abs_g.values * 1e3
        amp_e = ds_q.IQ_abs_e.values * 1e3
    else:
        return

    ax.plot(x_plot, amp_g, ".", ms=4, color="C0", alpha=0.6, label="|g⟩")
    ax.plot(x_plot, amp_e, ".", ms=4, color="C1", alpha=0.6, label="|e⟩")

    if fp is not None and fp.success:
        x_fine = np.linspace(x_hz[0], x_hz[-1], 600)
        x_fine_plot = x_fine * 1e-9 if use_full_freq else x_fine * 1e-6
        cg = fp.f_g if use_full_freq else fp.center_g_hz
        ce = fp.f_e if use_full_freq else fp.center_e_hz

        if not np.isnan(fp.amplitude_g):
            ax.plot(x_fine_plot,
                    lorentzian_dip(x_fine, fp.amplitude_g, cg, fp.kappa_g_hz, fp.offset_g) * 1e3,
                    "-", color="C0", lw=1.5)
        if not np.isnan(fp.amplitude_e):
            ax.plot(x_fine_plot,
                    lorentzian_dip(x_fine, fp.amplitude_e, ce, fp.kappa_e_hz, fp.offset_e) * 1e3,
                    "-", color="C1", lw=1.5)

        ax.axvline(fp.f_g * 1e-9 if use_full_freq else fp.center_g_hz * 1e-6,
                   color="C0", ls="--", lw=1, label=f"f_g={fp.f_g*1e-9:.5f} GHz")
        ax.axvline(fp.f_e * 1e-9 if use_full_freq else fp.center_e_hz * 1e-6,
                   color="C1", ls="--", lw=1, label=f"f_e={fp.f_e*1e-9:.5f} GHz")
        ax.axvline(fp.f_optimal * 1e-9 if use_full_freq else (fp.f_optimal - (fp.f_g - fp.center_g_hz)) * 1e-6,
                   color="k", ls="-", lw=1.5, label=f"f_opt={fp.f_optimal*1e-9:.5f} GHz")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("|IQ| (mV)")
    ax.legend(fontsize=8)


def _plot_phase(ax, ds_q, fp):
    x_plot, x_hz, xlabel = _x_axis(ds_q)
    use_full_freq = "full_freq" in ds_q.coords

    if "phase" not in ds_q.data_vars:
        ax.set_visible(False)
        return

    if "qubit_state" in ds_q.dims:
        ph_g = ds_q.phase.sel(qubit_state=0).values
        ph_e = ds_q.phase.sel(qubit_state=1).values
    elif "phase_g" in ds_q.data_vars:
        ph_g = ds_q.phase_g.values
        ph_e = ds_q.phase_e.values
    else:
        ax.set_visible(False)
        return

    ax.plot(x_plot, ph_g, ".", ms=4, color="C0", alpha=0.6, label="|g⟩")
    ax.plot(x_plot, ph_e, ".", ms=4, color="C1", alpha=0.6, label="|e⟩")

    if fp is not None:
        x_fine = np.linspace(x_hz[0], x_hz[-1], 600)
        x_fine_plot = x_fine * 1e-9 if use_full_freq else x_fine * 1e-6
        cg_ph = fp.f_g if use_full_freq else fp.center_phase_g_hz
        ce_ph = fp.f_e if use_full_freq else fp.center_phase_e_hz

        if not np.isnan(fp.phi0_g) and not np.isnan(fp.kappa_phase_g_hz):
            ax.plot(x_fine_plot,
                    _phase_model(x_fine, fp.phi0_g, cg_ph, fp.kappa_phase_g_hz),
                    "-", color="C0", lw=1.5,
                    label=f"fit: κ={fp.kappa_phase_g_hz*2e-3:.1f} kHz")
        if not np.isnan(fp.phi0_e) and not np.isnan(fp.kappa_phase_e_hz):
            ax.plot(x_fine_plot,
                    _phase_model(x_fine, fp.phi0_e, ce_ph, fp.kappa_phase_e_hz),
                    "-", color="C1", lw=1.5,
                    label=f"fit: κ={fp.kappa_phase_e_hz*2e-3:.1f} kHz")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Phase (rad)")
    ax.legend(fontsize=8)
