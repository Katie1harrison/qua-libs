import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

try:
    from qualang_tools.analysis import two_state_discriminator
except ImportError:
    two_state_discriminator = None


@dataclass
class FitParameters:
    """Fit results for the readout length optimization."""

    optimal_readout_length_ns: int
    optimal_fidelity: float
    num_divisions: int
    fidelities: List[float]
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, r in fit_results.items():
        status = "SUCCESS!" if r["success"] else "FAIL!"
        log_callable(
            f"Readout length for qubit {q}: optimal = {r['optimal_readout_length_ns']} ns, "
            f"fidelity = {100 * r['optimal_fidelity']:.1f}% --> {status}"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Dataset is already in correct form (raw shots per division). Nothing to transform."""
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Compute readout fidelity at each cumulative readout length division using two_state_discriminator.

    Expects ds to contain per-qubit variables Ig, Qg, Ie, Qe with dims (n_runs, division).
    """
    if two_state_discriminator is None:
        raise ImportError("qualang_tools.analysis.two_state_discriminator is required.")

    qubits = node.namespace["qubits"]
    division_length_ns = node.parameters.division_length_in_ns

    fit_results = {}
    fidelity_data = {}

    for q in qubits:
        qname = q.name
        Ig = ds[f"Ig_{qname}"].values  # shape: (n_runs, n_divisions)
        Qg = ds[f"Qg_{qname}"].values
        Ie = ds[f"Ie_{qname}"].values
        Qe = ds[f"Qe_{qname}"].values

        n_divisions = Ig.shape[1]
        fidelities = []

        for d in range(n_divisions):
            # two_state_discriminator returns: angle, threshold, fidelity, gg, ge, eg, ee
            try:
                _, _, fidelity, *_ = two_state_discriminator(
                    Ig[:, d], Qg[:, d], Ie[:, d], Qe[:, d], b_print=False, b_plot=False
                )
                fidelities.append(float(fidelity))
            except Exception:
                fidelities.append(float("nan"))

        fidelities_arr = np.array(fidelities)
        valid = np.isfinite(fidelities_arr)

        if valid.any():
            best_div = int(np.nanargmax(fidelities_arr))
            optimal_length_ns = (best_div + 1) * division_length_ns
            optimal_fidelity = fidelities_arr[best_div]
            success = True
        else:
            optimal_length_ns = n_divisions * division_length_ns
            optimal_fidelity = float("nan")
            success = False

        fidelity_data[qname] = fidelities_arr
        fit_results[qname] = FitParameters(
            optimal_readout_length_ns=int(optimal_length_ns),
            optimal_fidelity=float(optimal_fidelity),
            num_divisions=n_divisions,
            fidelities=fidelities_arr.tolist(),
            success=success,
        )

    # Attach fidelity curves to dataset
    lengths_ns = np.arange(1, list(fit_results.values())[0].num_divisions + 1) * division_length_ns
    ds_fit = ds.assign(
        {
            "fidelity": xr.DataArray(
                np.array([fidelity_data[q.name] for q in qubits]),
                dims=["qubit", "readout_length_ns"],
                coords={
                    "qubit": [q.name for q in qubits],
                    "readout_length_ns": lengths_ns,
                },
                attrs={"long_name": "Readout fidelity", "units": ""},
            )
        }
    )

    return ds_fit, fit_results
