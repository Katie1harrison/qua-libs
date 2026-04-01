# %% {Imports}
import matplotlib.pyplot as plt
from dataclasses import asdict

import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
    get_qubits,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from calibration_utils.cavity_mode_chi import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)

# %% {Node initialisation}
description = """
        CAVITY MODE CHI (DISPERSIVE SHIFT)
Measures the dispersive shift chi between the qubit and a cavity mode (e.g. alice
or bob) by performing two interleaved qubit ge spectroscopy sweeps:

  - Branch 0 (|g,0>): qubit ge spectroscopy with cavity in vacuum.
  - Branch 1 (|g,1>): qubit ge spectroscopy after preparing Fock state |1> in
    the cavity via |f,0> → f0g1 π-pulse → |g,1>.

When the cavity has one photon, the qubit ge transition shifts by chi:
    chi = f_ge(0 photons) − f_ge(1 photon)

Both curves are fit to Lorentzians to extract the peak positions and chi.

Prerequisites:
    - Calibrated ge and ef pulses (nodes 04b, 13).
    - Calibrated f0g1 π-pulse (nodes 22 or 24).

Parameters:
    - mode_name:              'alice' or 'bob'
    - frequency_span_in_mhz: frequency span around qubit ge frequency [MHz]
    - frequency_step_in_mhz: frequency step [MHz]

State update:
    - cavity_mode.chi  [Hz]
"""


node = QualibrationNode[Parameters, Quam](
    name="26_cavity_mode_chi",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.frequency_span_in_mhz = 2.0
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
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    cavity_mode = _get_cavity_mode(node)

    # Sweep axes:
    #   outer dim = detuning (buffered second in stream_processing)
    #   inner dim = photon_branch [0, 1] (buffered first in stream_processing)
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            dfs, attrs={"long_name": "qubit detuning", "units": "Hz"}
        ),
        "photon_branch": xr.DataArray(
            [0, 1],
            attrs={"long_name": "photon branch (0=vacuum, 1=Fock1)"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        df = declare(int)
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

                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        # ── Branch 0: no photon — standard ge spectroscopy ────────
                        # Reset qubit thermally (also allows any leftover cavity
                        # photon from the previous shot's branch 1 to decay)
                        qubit.xy.wait(2 * qubit.thermalization_time * u.ns)

                        # Saturation pulse at swept ge frequency
                        qubit.xy.update_frequency(
                            df + qubit.xy.intermediate_frequency
                        )
                        qubit.xy.play("saturation")

                        # Measure
                        align(qubit.xy.name, qubit.resonator.name)
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])  # photon_branch=0 save
                        else:
                            qubit.resonator.measure(
                                "readout", qua_vars=(I[i], Q[i])
                            )
                            save(I[i], I_st[i])  # photon_branch=0 save
                            save(Q[i], Q_st[i])

                        qubit.resonator.wait(node.machine.depletion_time * u.ns)

                        # ── Branch 1: Fock1 photon — ge spectroscopy shifts by chi
                        # Second thermal reset to initialise cavity in vacuum
                        qubit.xy.wait(2 * qubit.thermalization_time * u.ns)

                        # Prepare qubit in |f>: π_ge then π_ef
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x180")  # |g> → |e>
                        qubit.xy.update_frequency(
                            qubit.xy.intermediate_frequency - qubit.anharmonicity
                        )
                        qubit.xy.play("x180")  # |e> → |f>

                        # Create photon: f0g1 π  |f,0> → |g,1>
                        align(qubit.xy.name, cavity_mode.cavity_mode_drive.name)
                        cavity_mode.cavity_mode_drive.play("f0g1_pi")

                        # Qubit is now in |g>, cavity has 1 photon.
                        # Play saturation pulse at swept ge frequency.
                        # The ge transition is shifted by chi so the peak
                        # will appear at f_ge + chi.
                        align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)
                        qubit.xy.update_frequency(
                            df + qubit.xy.intermediate_frequency
                        )
                        qubit.xy.play("saturation")

                        # Measure
                        align(qubit.xy.name, qubit.resonator.name)
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])  # photon_branch=1 save
                        else:
                            qubit.resonator.measure(
                                "readout", qua_vars=(I[i], Q[i])
                            )
                            save(I[i], I_st[i])  # photon_branch=1 save
                            save(Q[i], Q_st[i])

                        qubit.resonator.wait(node.machine.depletion_time * u.ns)

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    (
                        state_st[i]
                        .buffer(2)           # photon_branch (inner: b0, b1)
                        .buffer(len(dfs))    # detuning (outer)
                        .average()
                        .save(f"state{i + 1}")
                    )
                else:
                    (
                        I_st[i]
                        .buffer(2)
                        .buffer(len(dfs))
                        .average()
                        .save(f"I{i + 1}")
                    )
                    (
                        Q_st[i]
                        .buffer(2)
                        .buffer(len(dfs))
                        .average()
                        .save(f"Q{i + 1}")
                    )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_raw"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q_name: ("successful" if res["success"] else "failed")
        for q_name, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
    )
    plt.show()
    node.results["figures"] = {"cavity_mode_chi": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    mode_name = node.parameters.mode_name
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            chi_hz = node.results["fit_results"][q.name]["chi_hz"]
            for cav in node.machine.cavities.values():
                mode = getattr(cav, mode_name, None)
                if mode is not None and hasattr(mode, "chi"):
                    mode.chi = float(chi_hz)
                    break


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
