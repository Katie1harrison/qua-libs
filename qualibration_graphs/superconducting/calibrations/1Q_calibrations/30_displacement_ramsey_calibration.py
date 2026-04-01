# %% {Imports}
import logging
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_builder.tools.power_tools import calculate_voltage_scaling_factor
from qualibration_libs.core import tracked_updates
from quam_config import Quam
from calibration_utils.displacement_ramsey_calibration import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    plot_ramsey_traces,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

logger = logging.getLogger(__name__)


# %% {Description}
description = """
        DISPLACEMENT RAMSEY CALIBRATION (30)

Calibrates the mapping between cavity displacement pulse amplitude A and mean
photon number n̄ using the Ramsey photon-number method.

Sequence (per amplitude A and delay τ):
  1. Thermalize cavity (wait 5× cavity T1).
  2. Apply displacement pulse at amplitude_scale = A.
  3. Play x90 on qubit.
  4. Wait τ.
  5. Play x90 on qubit.
  6. Measure qubit state.

The resulting Ramsey signal for a cavity coherent state is:

    P_e(τ) = 0.5 · (1 + exp(-n̄·(1-cos(θ))) · cos(n̄·sin(θ)))

where θ = 2π·chi_hz·τ and chi_hz is the dispersive-shift peak spacing [Hz]
(calibrated by node 28 or 26).

For each amplitude A the trace is fitted to extract n̄(A), then the calibration
curve n̄ = k·A² is fitted to obtain:

    A₁ph = 1/√k  — amplitude_scale that deposits exactly 1 photon on average.

Prerequisites:
    - Calibrated x90 pulse on the qubit (nodes 04a/04b).
    - Known dispersive shift chi_hz (node 28 or 26), OR set node.parameters.chi_hz.
    - Displacement pulse operation present on cavity_mode_drive.

State updates:
    - cavity_mode.cavity_mode_drive.operations["displacement"].amplitude  (= A₁ph · base_amp)
    - cavity_transmon_pairs["{qubit}_{mode}"].displacement_k  (= k)
"""

node = QualibrationNode[Parameters, Quam](
    name="30_displacement_ramsey_calibration",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.chi_hz = 1.2e6  # Hz — dispersive shift peak spacing
    # node.parameters.amp_max = 1.0
    # node.parameters.tau_max_ns = 5000
    pass


node.machine = Quam.load()


def _get_cavity_mode(node):
    mode_name = node.parameters.mode_name
    for cav in node.machine.cavities.values():
        mode = getattr(cav, mode_name, None)
        if mode is not None:
            return mode
    raise KeyError(f"Cavity mode '{mode_name}' not found in machine.cavities")


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots

    # Displacement amplitude sweep (linear, in amplitude_scale units)
    amps = np.linspace(node.parameters.amp_min, node.parameters.amp_max, node.parameters.amp_points)

    # Ramsey delay sweep: ensure multiples of 4 ns (1 clock cycle)
    tau_min_clk = max(node.parameters.tau_min_ns // 4, 4)
    tau_max_clk = max(node.parameters.tau_max_ns // 4, tau_min_clk + 1)
    tau_clk = np.round(
        np.linspace(tau_min_clk, tau_max_clk, node.parameters.tau_points)
    ).astype(int)
    tau_ns = (4 * tau_clk).astype(float)

    cavity_mode = _get_cavity_mode(node)
    node.namespace["cavity_mode"] = cavity_mode

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp": xr.DataArray(
            amps, attrs={"long_name": "displacement amplitude scale", "units": "a.u."}
        ),
        "tau": xr.DataArray(
            tau_ns, attrs={"long_name": "Ramsey delay", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        a = declare(fixed)
        tau = declare(int)
        n_st = declare_stream()

        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        else:
            I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amps)):
                    with for_each_(tau, tau_clk.tolist()):
                        for i, qubit in multiplexed_qubits.items():
                            # Thermalize: wait T1 * factor of the cavity mode.
                            # T1 is in seconds; Channel.wait() takes clock cycles (1 clk = 4 ns).
                            therm_clk = int(min(max(cavity_mode.T1 * cavity_mode.thermalization_time_factor * 1e9 / 4, 4), 2_500_000_000))
                            cavity_mode.cavity_mode_drive.wait(therm_clk)
                            # qubit.xy.wait(therm_clk)

                            # Apply displacement
                            dur_ns = node.parameters.displacement_pulse_duration_ns
                            if dur_ns is not None:
                                cavity_mode.cavity_mode_drive.play(
                                    "displacement", duration=dur_ns // 4, amplitude_scale=a
                                )
                            else:
                                cavity_mode.cavity_mode_drive.play(
                                    "displacement", amplitude_scale=a
                                )

                            # Ramsey sequence: x90 – wait τ – x90
                            align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)
                            qubit.xy.play("x90")
                            qubit.xy.wait(tau)
                            qubit.xy.play("x90")

                            # Measure
                            align(qubit.xy.name, qubit.resonator.name)
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure(
                                    "readout", qua_vars=(I[i], Q[i])
                                )
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

                            qubit.resonator.wait(node.machine.depletion_time * u.ns)
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(tau_clk)).buffer(len(amps)).average().save(
                        f"state{i + 1}"
                    )
                else:
                    I_st[i].buffer(len(tau_clk)).buffer(len(amps)).average().save(
                        f"I{i + 1}"
                    )
                    Q_st[i].buffer(len(tau_clk)).buffer(len(amps)).average().save(
                        f"Q{i + 1}"
                    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)
    node.namespace["cavity_mode"] = _get_cavity_mode(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_raw"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q: ("successful" if res["success"] else "failed")
        for q, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    figures = {}
    figures["displacement_ramsey_2d"] = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
    )
    plt.show()

    figures["displacement_ramsey_traces"] = plot_ramsey_traces(
        node.results["ds_raw"],
        node.namespace["qubits"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
    )
    plt.show()

    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    cavity_mode = node.namespace["cavity_mode"]
    mode_name = node.parameters.mode_name

    # Read the base amplitude before any tracked-update context (= operation amplitude as-is)
    base_amp = float(cavity_mode.cavity_mode_drive.operations["displacement"].amplitude)

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            res = node.results["fit_results"].get(qubit.name)
            if res is None or not res["success"]:
                continue

            amp_one_scale = res["amp_for_one_photon"]  # scale factor relative to base_amp
            k = res["k"]

            # Store calibrated absolute amplitude: play at scale=1.0 → 1 photon on average
            cal_amplitude = base_amp * amp_one_scale
            cavity_mode.cavity_mode_drive.operations["displacement"].amplitude = float(cal_amplitude)

            # Update CavityTransmonPair (create on-the-fly if missing)
            pair_key = f"{qubit.name}_{mode_name}"
            pairs = getattr(node.machine, "cavity_transmon_pairs", None)
            if pairs is not None:
                if pair_key not in pairs:
                    from quam_builder.architecture.superconducting.qubit_pair import CavityTransmonPair
                    pairs[pair_key] = CavityTransmonPair(
                        qubit_name=qubit.name, cavity_mode_name=mode_name
                    )
                    logger.info(f"Created CavityTransmonPair '{pair_key}' on the fly.")
                if pair_key in pairs:
                    pairs[pair_key].displacement_k = float(k)
                    # Also persist the fitted chi so future nodes can use it
                    chi_fitted = res.get("chi_hz")
                    if chi_fitted is not None and np.isfinite(chi_fitted) and chi_fitted > 0:
                        if hasattr(pairs[pair_key], "chi"):
                            pairs[pair_key].chi = float(chi_fitted)
            else:
                logger.warning(
                    f"machine has no cavity_transmon_pairs field — "
                    f"k for '{pair_key}' not persisted to QUAM."
                )

            break  # one cavity mode shared across all qubits in this run


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
