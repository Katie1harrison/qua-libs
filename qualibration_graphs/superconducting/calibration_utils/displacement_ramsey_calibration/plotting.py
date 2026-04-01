"""Plotting for displacement Ramsey calibration (node 30).

Produces two subplots per qubit:
  1. 2D colormesh: Ramsey delay τ (x) vs displacement amplitude A (y) vs state (colour).
     Overlays the fitted Ramsey curve at each amplitude as dashed lines.
  2. n̄ vs A: raw (A, n̄) scatter + fitted n̄ = k·A² curve + vertical marker at A₁ph.
"""
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
from matplotlib.figure import Figure

from calibration_utils.displacement_ramsey_calibration.analysis import ramsey_photon_model


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    qubits,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
) -> Figure:
    """Plot Ramsey traces (2D map + n̄ calibration curve) for all qubits.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with dims ``qubit``, ``amp``, ``tau``.
    qubits : QubitCollection
        Qubit objects from ``node.namespace["qubits"]``.
    fit_results : dict, optional
        Dict mapping qubit name → FitParameters dict (already ``asdict``-converted).
    mode_name : str
        Cavity mode label for plot titles.
    """
    num_qubits = len(list(qubits))
    fig = plt.figure(figsize=(14, 5 * num_qubits))
    outer = gridspec.GridSpec(num_qubits, 1, figure=fig, hspace=0.55)

    for qubit_idx, qubit in enumerate(qubits):
        q_name = qubit.name
        ds_q = ds.sel(qubit=q_name)

        signal_name = "state" if "state" in ds_q.data_vars else "I"
        if signal_name == "state":
            data = ds_q.state.values          # shape (n_amps, n_tau)
            clabel = "State population"
        else:
            data = ds_q.I.values * 1e3        # mV
            clabel = "I (mV)"

        amps = ds_q.amp.values               # shape (n_amps,)
        tau_ns = ds_q.tau.values             # shape (n_tau,), in ns

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[qubit_idx], wspace=0.38
        )
        ax2d = fig.add_subplot(inner[0])
        ax_nb = fig.add_subplot(inner[1])

        # ── 2D colormesh ──────────────────────────────────────────────────────
        pcm = ax2d.pcolormesh(
            tau_ns * 1e-3, amps, data, shading="auto", cmap="RdBu_r", vmin=0, vmax=1
        )
        fig.colorbar(pcm, ax=ax2d, label=clabel)
        ax2d.set_xlabel("Ramsey delay τ (µs)")
        ax2d.set_ylabel("Displacement amplitude scale A")
        ax2d.set_title(f"{q_name} — {mode_name}: Ramsey traces vs amplitude")

        # Overlay fitted Ramsey curves per amplitude
        if fit_results and q_name in fit_results:
            res = fit_results[q_name]
            if res["success"] and np.isfinite(res["chi_hz"]) and res["chi_hz"] > 0:
                chi_hz = res["chi_hz"]
                k = res["k"]
                tau_fine = np.linspace(tau_ns.min(), tau_ns.max(), 300)
                for amp_idx, A in enumerate(amps):
                    nbar = k * A ** 2
                    y_fit = ramsey_photon_model(tau_fine, nbar, chi_hz)
                    ax2d.plot(
                        tau_fine * 1e-3, np.full_like(tau_fine, A),
                        color="gray", lw=0.3, alpha=0.0   # invisible anchor line
                    )
                    # Draw as a coloured trace overlaid in data coordinates
                    ax2d.plot(
                        tau_fine * 1e-3,
                        A + (y_fit - 0.5) * (amps[1] - amps[0] if len(amps) > 1 else 0.05),
                        color="white", lw=0.8, alpha=0.6,
                    )

        # ── n̄ vs A ────────────────────────────────────────────────────────────
        if fit_results and q_name in fit_results:
            res = fit_results[q_name]
            if res["nbar_vs_amp"]:
                A_raw = np.array([v[0] for v in res["nbar_vs_amp"]])
                nb_raw = np.array([v[1] for v in res["nbar_vs_amp"]])
                ax_nb.scatter(A_raw, nb_raw, s=40, color="C0", zorder=5, label="data")

            if res["success"]:
                A_fit = np.linspace(0, float(amps.max()), 200)
                k = res["k"]
                ax_nb.plot(
                    A_fit, k * A_fit ** 2, color="C1", lw=2,
                    label=f"k·A²  (k={k:.3f})"
                )
                ax_nb.axvline(
                    res["amp_for_one_photon"], color="C2", ls="--", lw=1.5,
                    label=f"A₁ph = {res['amp_for_one_photon']:.4f}"
                )
            else:
                ax_nb.set_title("FIT FAILED", color="red", fontsize=9)

        ax_nb.set_xlabel("Displacement amplitude scale A")
        ax_nb.set_ylabel("Mean photon number n̄")
        ax_nb.set_title(f"{q_name} — n̄ vs A")
        ax_nb.legend(fontsize=8)
        ax_nb.set_xlim(left=0)
        ax_nb.set_ylim(bottom=0)

    fig.suptitle(
        f"Displacement Ramsey Calibration — {mode_name} (30)", fontsize=12, y=1.01
    )
    fig.tight_layout()
    return fig


def plot_ramsey_traces(
    ds: xr.Dataset,
    qubits,
    fit_results: Optional[Dict] = None,
    mode_name: str = "alice",
    max_amps_to_show: int = 6,
) -> Figure:
    """Plot individual Ramsey traces (one per selected amplitude) with model overlay.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset with dims ``qubit``, ``amp``, ``tau``.
    qubits : QubitCollection
        Qubit objects from ``node.namespace["qubits"]``.
    fit_results : dict, optional
        Dict mapping qubit name → FitParameters dict.
    mode_name : str
        Cavity mode label for plot titles.
    max_amps_to_show : int
        Maximum number of amplitude traces to show (evenly selected from the sweep).
    """
    num_qubits = len(list(qubits))
    fig, axes = plt.subplots(
        num_qubits, max_amps_to_show,
        figsize=(3.5 * max_amps_to_show, 3.5 * num_qubits),
        squeeze=False,
    )

    for qubit_idx, qubit in enumerate(qubits):
        q_name = qubit.name
        ds_q = ds.sel(qubit=q_name)

        signal_name = "state" if "state" in ds_q.data_vars else "I"
        amps = ds_q.amp.values
        tau_ns = ds_q.tau.values

        # Select evenly spaced subset of amplitudes
        n_amps = len(amps)
        sel_indices = np.round(np.linspace(0, n_amps - 1, min(max_amps_to_show, n_amps))).astype(int)

        for col_idx, amp_idx in enumerate(sel_indices):
            ax = axes[qubit_idx, col_idx]
            A = float(amps[amp_idx])

            if signal_name == "state":
                signal = ds_q.isel(amp=amp_idx).state.values.astype(float)
                ylabel = "P(e)"
            else:
                signal = ds_q.isel(amp=amp_idx).I.values.astype(float) * 1e3
                ylabel = "I (mV)"

            ax.plot(tau_ns * 1e-3, signal, "o-", ms=3, lw=1.0, color="C0")

            # Overlay model
            if (
                fit_results and q_name in fit_results
                and fit_results[q_name]["success"]
                and signal_name == "state"
            ):
                res = fit_results[q_name]
                chi_hz = res["chi_hz"]
                nbar = res["k"] * A ** 2
                tau_fine = np.linspace(tau_ns.min(), tau_ns.max(), 300)
                y_model = ramsey_photon_model(tau_fine, nbar, chi_hz)
                ax.plot(
                    tau_fine * 1e-3, y_model, lw=1.5, color="C1",
                    label=f"n̄={nbar:.2f}"
                )
                ax.legend(fontsize=7, loc="upper right")

            ax.set_xlabel("τ (µs)")
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.set_title(f"A={A:.3f}", fontsize=9)
            ax.set_ylim(-0.05, 1.05)

        # Hide unused axes
        for col_idx in range(len(sel_indices), max_amps_to_show):
            axes[qubit_idx, col_idx].set_visible(False)

    fig.suptitle(
        f"Ramsey Traces — {mode_name} (30) — {q_name}", fontsize=11
    )
    fig.tight_layout()
    return fig
