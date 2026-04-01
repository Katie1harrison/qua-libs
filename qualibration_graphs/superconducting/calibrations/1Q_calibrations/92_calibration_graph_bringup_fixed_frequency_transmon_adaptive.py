# %%
"""
Fixed-Frequency Transmon Bring-Up Graph (Adaptive)

Full automated bring-up sequence for a fixed-frequency transmon qubit.

  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. mixer_calibration                                               │
  │  2. resonator_bringup (subgraph):                                   │
  │       resonator_discovery [loop: retry on no dip]:                  │
  │         broad_resonator_spectroscopy                                │
  │         ──► resonator_spectroscopy_high_power                       │
  │       ──► resonator_punch_out        [loop: retry on failure]       │
  │       ──► resonator_spectroscopy_low_power                          │
  │  3. qubit_calibration (subgraph, nested loops):                     │
  │       qubit_spectroscopy_vs_power    [inner loop: span expansion]   │
  │       ──► qubit_spectroscopy                                        │
  │       ──► power_rabi                 [inner loop: amp rescaling]    │
  │       [outer loop: restart on NO_OSCILLATION → new freq search]     │
  │  4. x180_fine_calibration (subgraph):                               │
  │       rabi_ramsey [loop: repeat until freq converges]               │
  │         power_rabi ──► ramsey                                       │
  │  5. T1                                                              │
  │  6. readout_frequency_optimization                                  │
  │  7. readout_power_optimization                                      │
  └─────────────────────────────────────────────────────────────────────┘
"""

from typing import List, Optional

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
)
from calibration_utils.bringup_graphs import (
    build_resonator_bringup,
    build_qubit_calibration,
    build_x180_fine_calibration,
    should_restart_qubit_calibration,
)

library = QualibrationLibrary.get_active_library()

test_qubits = ["q2"]


# ─── Top-level parameters ─────────────────────────────────────────────────────

class TransmonBringUpParameters(GraphParameters):
    """Parameters for the full fixed-frequency transmon bring-up graph."""

    qubits: List[str] = test_qubits
    multiplexed: bool = False

    # ── Iteration limits ──────────────────────────────────────────────────────
    max_resonator_discovery_iterations: int = 5
    max_punch_out_iterations: int = 5
    max_spec_vs_power_iterations: int = 5
    max_rabi_amp_iterations: int = 5
    max_qubit_calibration_iterations: int = 3

    # ── Mixer calibration ──────────────────────────────────────────────────────
    mixer_calibrate_resonator: bool = True
    mixer_calibrate_drive: bool = True

    # ── Resonator – broad spectroscopy ────────────────────────────────────────
    broad_frequency_span_mhz: float = 200.0
    broad_frequency_step_mhz: float = 0.1
    broad_num_shots: int = 50
    broad_peak_prominence: float = 2.0
    broad_peak_width: tuple = (1, 10.0)
    broad_readout_power_dbm: Optional[float] = 0.0
    broad_max_amp: float = 0.1
    blacklist_exclusion_radius_mhz: float = 10.0

    # ── Resonator – high-power confirmation ───────────────────────────────────
    high_power_frequency_span_mhz: float = 2.0
    high_power_frequency_step_mhz: float = 0.01
    high_power_num_shots: int = 100
    high_power_readout_power_dbm: Optional[float] = 0.0
    high_power_max_amp: float = 0.1

    # ── Resonator – punch-out ─────────────────────────────────────────────────
    punch_out_frequency_span_mhz: float = 2.0
    punch_out_frequency_step_mhz: float = 0.05
    punch_out_min_power_dbm: int = -30
    punch_out_max_power_dbm: int = 0
    punch_out_num_power_points: int = 2
    punch_out_max_amp: float = 0.1
    punch_out_num_shots: int = 100
    punch_out_frequency_shift_threshold_hz: float = 0.1e6
    use_adaptive_span: bool = True

    # ── Resonator – low-power fine spectroscopy ───────────────────────────────
    low_power_frequency_span_mhz: float = 2.0
    low_power_frequency_step_mhz: float = 0.001
    low_power_num_shots: int = 100
    low_power_readout_power_dbm: Optional[float] = None
    low_power_max_amp: float = 0.1

    # ── Resonator – shared ────────────────────────────────────────────────────

    # ── Qubit spectroscopy vs power ───────────────────────────────────────────
    spec_vs_power_frequency_span_mhz: float = 200.0
    spec_vs_power_frequency_step_mhz: float = 2.0
    spec_vs_power_num_power_points: int = 10
    spec_vs_power_num_shots: int = 100
    spec_vs_power_min_power_dbm: int = -80
    spec_vs_power_max_power_dbm: int = 0
    spec_vs_power_operation: str = "saturation"
    spec_vs_power_operation_len_ns: int = 200_000
    spec_vs_power_linewidth_threshold_hz: float = 10e6
    spec_vs_power_max_amplitude_opx: float = 0.24
    spec_vs_power_power_buffer_db: float = 3.0

    # ── Qubit spectroscopy (fine) ──────────────────────────────────────────────
    qubit_spec_frequency_span_mhz: float = 50.0
    qubit_spec_frequency_step_mhz: float = 0.1
    qubit_spec_operation_len_ns: int = 200_000
    qubit_spec_operation_amplitude_factor: float = 1.0
    qubit_spec_num_shots: int = 100

    # ── Power Rabi ────────────────────────────────────────────────────────────
    rabi_min_amp_factor: float = 0.001
    rabi_max_amp_factor: float = 1.5
    rabi_amp_factor_step: float = 0.005
    rabi_num_shots: int = 50

    # ── X180 fine calibration ─────────────────────────────────────────────────
    x180_rabi_min_amp_factor: float = 0.001
    x180_rabi_max_amp_factor: float = 1.99
    x180_rabi_amp_factor_step: float = 0.005
    x180_rabi_num_shots: int = 50
    x180_rabi_max_number_pulses_per_sweep: int = 1
    x180_ramsey_num_shots: int = 100
    x180_ramsey_frequency_detuning_in_mhz: float = 1.0
    x180_ramsey_max_wait_time_in_ns: int = 10_000
    x180_ramsey_wait_time_num_points: int = 200
    x180_ramsey_log_or_linear_sweep: str = "linear"
    x180_freq_threshold_hz: float = 50_000.0
    x180_max_iterations: int = 10

    # ── T1 ────────────────────────────────────────────────────────────────────
    t1_num_shots: int = 1000
    t1_min_wait_time_ns: int = 16
    t1_max_wait_time_ns: int = 200_000
    t1_wait_time_num_points: int = 100
    t1_log_or_linear_sweep: str = "log"

    # ── Readout frequency optimization ────────────────────────────────────────
    readout_freq_frequency_span_mhz: float = 2.0
    readout_freq_frequency_step_mhz: float = 0.01
    readout_freq_num_shots: int = 100

    # ── Readout power optimization ────────────────────────────────────────────
    readout_power_num_shots: int = 2000
    readout_power_start_amp: float = 0.5
    readout_power_end_amp: float = 1.5
    readout_power_num_amps: int = 10


# ─── Graph construction ───────────────────────────────────────────────────────

with QualibrationGraph.build(
    "transmon_bringup",
    parameters=TransmonBringUpParameters(),
) as graph:

    # ── 1. Mixer calibration ──────────────────────────────────────────────────
    mixer_calibration = library.nodes["01a_mixer_calibration"].copy(
        name="mixer_calibration",
        calibrate_resonator=graph.parameters.mixer_calibrate_resonator,
        calibrate_drive=graph.parameters.mixer_calibrate_drive,
    )
    graph.add_node(mixer_calibration)

    # ── 2. Resonator bring-up ─────────────────────────────────────────────────
    resonator_bringup = build_resonator_bringup(graph, library)
    graph.add_node(resonator_bringup)

    # ── 3. Qubit calibration ──────────────────────────────────────────────────
    qubit_calibration = build_qubit_calibration(graph, library)
    graph.add_node(qubit_calibration)
    graph.loop(
        qubit_calibration,
        on=should_restart_qubit_calibration,
        max_iterations=graph.parameters.max_qubit_calibration_iterations,
    )

    # ── 4. X180 fine calibration ──────────────────────────────────────────────
    x180_fine_calibration = build_x180_fine_calibration(graph, library)
    graph.add_node(x180_fine_calibration)

    # ── 5. T1 ─────────────────────────────────────────────────────────────────
    t1 = library.nodes["05_T1"].copy(
        name="T1",
        num_shots=graph.parameters.t1_num_shots,
        min_wait_time_in_ns=graph.parameters.t1_min_wait_time_ns,
        max_wait_time_in_ns=graph.parameters.t1_max_wait_time_ns,
        wait_time_num_points=graph.parameters.t1_wait_time_num_points,
        log_or_linear_sweep=graph.parameters.t1_log_or_linear_sweep,
    )
    graph.add_node(t1)

    # ── 6. Readout frequency optimization ─────────────────────────────────────
    readout_freq_opt = library.nodes["08a_readout_frequency_optimization"].copy(
        name="readout_frequency_optimization",
        multiplexed=graph.parameters.multiplexed,
        num_shots=graph.parameters.readout_freq_num_shots,
        frequency_span_in_mhz=graph.parameters.readout_freq_frequency_span_mhz,
        frequency_step_in_mhz=graph.parameters.readout_freq_frequency_step_mhz,
    )
    graph.add_node(readout_freq_opt)

    # ── 7. Readout power optimization ─────────────────────────────────────────
    readout_power_opt = library.nodes["08b_readout_power_optimization"].copy(
        name="readout_power_optimization",
        num_shots=graph.parameters.readout_power_num_shots,
        start_amp=graph.parameters.readout_power_start_amp,
        end_amp=graph.parameters.readout_power_end_amp,
        num_amps=graph.parameters.readout_power_num_amps,
    )
    graph.add_node(readout_power_opt)

    # ── Execution order ────────────────────────────────────────────────────────
    graph.connect(mixer_calibration, resonator_bringup)
    graph.connect(resonator_bringup, qubit_calibration)
    graph.connect(qubit_calibration, x180_fine_calibration)
    graph.connect(x180_fine_calibration, t1)
    graph.connect(t1, readout_freq_opt)
    graph.connect(readout_freq_opt, readout_power_opt)


graph.run()
