"""Plotting utilities for the GEF dispersive shift measurement."""
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
    """Plot amplitude and phase spectra for |g⟩, |e⟩, |f⟩ with Lorentzian / arctan fits."""
    n_qubits = len(list(qubits))
    fig, axes = plt.subplots(2, n_qubits, figsize=(7 * n_qubits, 8), squeeze=False)

    for col, q_obj in enumerate(qubits):
        q_name = q_obj.name
        fp = fits.get(q_name) if isinstance(fits, dict) else None
        ds_q = ds.sel(qubit=q_name)
        _plot_amplitude(axes[0, col], ds_q, fp)
        _plot_phase(axes[1, col], ds_q, fp)
        axes[0, col].set_title(f"{q_name}" + (
            f"\nchi_ge={fp.chi_ge_hz*1e-3:.1f} kHz  chi_ef={fp.chi_ef_hz*1e-3:.1f} kHz\n"
            f"κ (FWHM): g={fp.kappa_g_hz*2e-3:.1f} e={fp.kappa_e_hz*2e-3:.1f} f={fp.kappa_f_hz*2e-3:.1f} kHz"
            if fp is not None and fp.success else ""
        ))

    fig.suptitle("GEF dispersive shift — |g⟩, |e⟩, |f⟩ resonator spectra")
    fig.tight_layout()
    return fig


def _x_axis(ds_q):
    if "full_freq" in ds_q.coords:
        x_hz = ds_q.full_freq.values
        return x_hz * 1e-9, x_hz, "RF frequency (GHz)"
    else:
        x_hz = ds_q.detuning.values
        return x_hz * 1e-6, x_hz, "Detuning (MHz)"


def _plot_amplitude(ax, ds_q, fp):
    x_plot, x_hz, xlabel = _x_axis(ds_q)
    use_full_freq = "full_freq" in ds_q.coords

    amp_g = ds_q.IQ_abs.sel(qubit_state=0).values * 1e3
    amp_e = ds_q.IQ_abs.sel(qubit_state=1).values * 1e3
    amp_f = ds_q.IQ_abs.sel(qubit_state=2).values * 1e3

    ax.plot(x_plot, amp_g, ".", ms=4, color="C0", alpha=0.6, label="|g⟩")
    ax.plot(x_plot, amp_e, ".", ms=4, color="C1", alpha=0.6, label="|e⟩")
    ax.plot(x_plot, amp_f, ".", ms=4, color="C2", alpha=0.6, label="|f⟩")

    if fp is not None and fp.success:
        x_fine = np.linspace(x_hz[0], x_hz[-1], 600)
        x_fine_plot = x_fine * 1e-9 if use_full_freq else x_fine * 1e-6

        for center_attr, amp_attr, kappa_attr, offset_attr, color, freq_attr in [
            ("f_g" if use_full_freq else "center_g_hz", "amplitude_g", "kappa_g_hz", "offset_g", "C0", "f_g"),
            ("f_e" if use_full_freq else "center_e_hz", "amplitude_e", "kappa_e_hz", "offset_e", "C1", "f_e"),
            ("f_f" if use_full_freq else "center_f_hz", "amplitude_f", "kappa_f_hz", "offset_f", "C2", "f_f"),
        ]:
            c = getattr(fp, center_attr.split(" if ")[0].split(" else ")[0] if " if " in center_attr else center_attr)
            a = getattr(fp, amp_attr)
            k = getattr(fp, kappa_attr)
            o = getattr(fp, offset_attr)
            if not np.isnan(a):
                ax.plot(x_fine_plot, lorentzian_dip(x_fine, a, c, k, o) * 1e3,
                        "-", color=color, lw=1.5)
            freq = getattr(fp, freq_attr)
            ax.axvline(freq * 1e-9 if use_full_freq else (freq - fp.f_g + fp.center_g_hz) * 1e-6,
                       color=color, ls="--", lw=1)

        f_opt_x = fp.f_optimal * 1e-9 if use_full_freq else (fp.f_optimal - fp.f_g + fp.center_g_hz) * 1e-6
        ax.axvline(f_opt_x, color="k", ls="-", lw=2,
                   label=f"f_opt (|e⟩): {fp.f_optimal*1e-9:.5f} GHz")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("|IQ| (mV)")
    ax.legend(fontsize=8)


def _plot_phase(ax, ds_q, fp):
    x_plot, x_hz, xlabel = _x_axis(ds_q)
    use_full_freq = "full_freq" in ds_q.coords

    if "phase" not in ds_q.data_vars:
        ax.set_visible(False)
        return

    ph_g = ds_q.phase.sel(qubit_state=0).values
    ph_e = ds_q.phase.sel(qubit_state=1).values
    ph_f = ds_q.phase.sel(qubit_state=2).values

    ax.plot(x_plot, ph_g, ".", ms=4, color="C0", alpha=0.6, label="|g⟩")
    ax.plot(x_plot, ph_e, ".", ms=4, color="C1", alpha=0.6, label="|e⟩")
    ax.plot(x_plot, ph_f, ".", ms=4, color="C2", alpha=0.6, label="|f⟩")

    if fp is not None:
        x_fine = np.linspace(x_hz[0], x_hz[-1], 600)
        x_fine_plot = x_fine * 1e-9 if use_full_freq else x_fine * 1e-6

        for phi0, center, kappa, color, label in [
            (fp.phi0_g, fp.f_g if use_full_freq else fp.center_phase_g_hz,
             fp.kappa_phase_g_hz, "C0", f"κ_g={fp.kappa_phase_g_hz*2e-3:.1f} kHz"),
            (fp.phi0_e, fp.f_e if use_full_freq else fp.center_phase_e_hz,
             fp.kappa_phase_e_hz, "C1", f"κ_e={fp.kappa_phase_e_hz*2e-3:.1f} kHz"),
            (fp.phi0_f, fp.f_f if use_full_freq else fp.center_phase_f_hz,
             fp.kappa_phase_f_hz, "C2", f"κ_f={fp.kappa_phase_f_hz*2e-3:.1f} kHz"),
        ]:
            if not np.isnan(phi0) and not np.isnan(kappa):
                ax.plot(x_fine_plot, _phase_model(x_fine, phi0, center, kappa),
                        "-", color=color, lw=1.5, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Phase (rad)")
    ax.legend(fontsize=8)
