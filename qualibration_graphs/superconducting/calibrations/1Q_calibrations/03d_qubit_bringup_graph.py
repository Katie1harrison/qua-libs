# %%
"""
Qubit Optimization Graph (Adaptive)

This graph finds the ideal qubit parameters through a self-correcting nested loop.
The three nodes form an inner calibration subgraph that is repeated until the Rabi
calibration succeeds:

  ┌─ outer loop (restart on NO_OSCILLATION) ──────────────────────────────────┐
  │  spec_vs_power ──► qubit_spec ──► power_rabi                              │
  │  [ inner loop ]                   [ inner loop ]                          │
  │  (span expansion)                 (amplitude rescaling)                   │
  └────────────────────────────────────────────────────────────────────────────┘

Retry logic:
  - power_rabi TOO_MANY / TOO_FEW periods → only power_rabi is retried after
    the adaptive amplitude rescaling (new_amp = old_amp / num_periods).
  - power_rabi NO_OSCILLATION → current qubit frequency is blacklisted in
    temp_calibration and the outer loop restarts from spec_vs_power to find
    a new candidate frequency.
"""

from typing import List

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
)
from calibration_utils.bringup_graphs import (
    build_qubit_calibration,
    should_restart_qubit_calibration,
)

library = QualibrationLibrary.get_active_library()

test_qubits = ["q1"]


class QubitOptimizationParameters(GraphParameters):
    """Parameters for the adaptive qubit optimization graph."""

    qubits: List[str] = test_qubits

    # Iteration limits
    max_spec_vs_power_iterations: int = 5
    max_rabi_amp_iterations: int = 5
    max_qubit_calibration_iterations: int = 3

    # General
    multiplexed: bool = False

    # Qubit spectroscopy vs power
    spec_vs_power_frequency_span_mhz: float = 200
    spec_vs_power_frequency_step_mhz: float = 2
    spec_vs_power_num_power_points: int = 10
    spec_vs_power_num_shots: int = 100
    spec_vs_power_min_power_dbm: int = -80
    spec_vs_power_max_power_dbm: int = 0
    spec_vs_power_operation: str = "saturation"
    spec_vs_power_operation_len_ns: int = 200_000
    spec_vs_power_linewidth_threshold_hz: float = 10e6
    spec_vs_power_max_amplitude_opx: float = 0.24
    spec_vs_power_min_peak_fraction: float = 0.3
    spec_vs_power_power_buffer_db: float = 3.0

    # Standard qubit spectroscopy
    qubit_spec_frequency_span_mhz: float = 50
    qubit_spec_frequency_step_mhz: float = 0.1
    qubit_spec_operation_len_ns: int = 200_000
    qubit_spec_operation_amplitude_factor: float = 1.0
    qubit_spec_num_shots: int = 100
    qubit_spec_min_peak_fraction: float = 0.3

    # Power Rabi
    rabi_min_amp_factor: float = 0.001
    rabi_max_amp_factor: float = 1.99
    rabi_amp_factor_step: float = 0.005
    rabi_num_shots: int = 50


with QualibrationGraph.build(
    "qubit_optimization",
    parameters=QubitOptimizationParameters(),
) as graph:
    qubit_calibration = build_qubit_calibration(graph, library)
    graph.add_node(qubit_calibration)
    graph.loop(
        qubit_calibration,
        on=should_restart_qubit_calibration,
        max_iterations=graph.parameters.max_qubit_calibration_iterations,
    )

graph.run()
