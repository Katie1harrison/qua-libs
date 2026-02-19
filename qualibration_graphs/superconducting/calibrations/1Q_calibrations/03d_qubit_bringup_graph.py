# %% {Qubit Bring-up}
"""
Qubit Bring-Up Graph (Adaptive)

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

import logging
from typing import List

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
    QualibrationNode,
)
from calibration_utils.error_codes import PowerRabiErrorCode

library = QualibrationLibrary.get_active_library()

logger = logging.getLogger(__name__)

MAX_SPEC_ITERATIONS = 5       # adaptive retries for spec_vs_power (span expansion)
MAX_RABI_AMP_ITERATIONS = 5   # retries of power_rabi for amplitude-only adjustment
MAX_OUTER_ITERATIONS = 3      # full-sequence restarts (NO_OSCILLATION → new freq search)


class QubitBringupParameters(GraphParameters):
    """Parameters for the adaptive qubit optimization graph."""

    qubits: List[str] = ["q1"]

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
    spec_vs_power_max_amplitude_opx: float = 0.1
    spec_vs_power_min_peak_fraction: float = 0.3

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


class QubitCalibrationSubgraphParameters(GraphParameters):
    """Minimal parameters for the inner calibration subgraph."""

    qubits: List[str] = ["q0"] #QubitBringupParameters.qubits


# ─── Condition functions ─────────────────────────────────────────────────────


def should_repeat_spec_vs_power(node: QualibrationNode, target: str) -> bool:
    """Retry spec_vs_power if no peak was found.

    The adaptive span/power logic in the node's update_state automatically
    expands the search range before the next iteration.
    """
    if node.outcomes.get(target) == "failed":
        logger.info(f"{target}: Spectroscopy vs power failed; retrying with expanded span.")
        return True
    logger.info(f"{target}: Spectroscopy vs power succeeded.")
    return False


def should_repeat_rabi_amplitude(node: QualibrationNode, target: str) -> bool:
    """Retry power_rabi only when the failure is an amplitude mismatch.

    TOO_MANY_PERIODS or TOO_FEW_PERIODS: the adaptive update_state has already
    rescaled the base amplitude (new_amp = old_amp / num_periods) so the next
    iteration will converge toward ~1 oscillation period.

    NO_OSCILLATION: do NOT retry here – escalate to the outer loop so the full
    frequency-search sequence can restart with the current frequency blacklisted.
    """
    if node.outcomes.get(target) == "failed":
        error_code = (
            node.results.get("fit_results", {})
            .get(target, {})
            .get("error_code", int(PowerRabiErrorCode.SUCCESS))
        )
        if error_code in (
            int(PowerRabiErrorCode.TOO_MANY_PERIODS),
            int(PowerRabiErrorCode.TOO_FEW_PERIODS),
        ):
            logger.info(
                f"{target}: Rabi amplitude mismatch "
                f"({PowerRabiErrorCode(error_code).name}); "
                "retrying with rescaled base amplitude."
            )
            return True
        logger.info(
            f"{target}: Rabi failed ({PowerRabiErrorCode(error_code).name}); "
            "escalating to full frequency search."
        )
    return False


def should_restart_qubit_calibration(node, target: str) -> bool:
    """Restart the full calibration subgraph if it failed.

    This handles the NO_OSCILLATION case: the qubit frequency is now blacklisted
    in temp_calibration, so the next spec_vs_power run will avoid it.
    """
    if node.outcomes.get(target) == "failed":
        logger.info(
            f"{target}: Qubit calibration failed; restarting frequency search."
        )
        return True
    logger.info(f"{target}: Qubit calibration succeeded.")
    return False


# ─── Graph construction ──────────────────────────────────────────────────────

with QualibrationGraph.build(
    "qubit_optimization",
    parameters=QubitBringupParameters(),
) as graph:

    # Inner calibration subgraph: spec_vs_power → qubit_spec → power_rabi
    # This subgraph is restarted by the outer loop when power_rabi signals
    # NO_OSCILLATION (wrong qubit frequency → need a new frequency search).
    with QualibrationGraph.build(
        "qubit_calibration",
        parameters=QubitCalibrationSubgraphParameters(),
    ) as qubit_calibration:

        # 1. Qubit spectroscopy vs power – find qubit frequency & optimal drive power
        spec_vs_power = library.nodes["03c_qubit_spectroscopy_vs_power"].copy(
            name="qubit_spectroscopy_vs_power",
            use_adaptive_span=True,
            multiplexed=graph.parameters.multiplexed,
            frequency_span_in_mhz=graph.parameters.spec_vs_power_frequency_span_mhz,
            frequency_step_in_mhz=graph.parameters.spec_vs_power_frequency_step_mhz,
            num_power_points=graph.parameters.spec_vs_power_num_power_points,
            num_shots=graph.parameters.spec_vs_power_num_shots,
            min_power_dbm=graph.parameters.spec_vs_power_min_power_dbm,
            max_power_dbm=graph.parameters.spec_vs_power_max_power_dbm,
            operation=graph.parameters.spec_vs_power_operation,
            operation_len_in_ns=graph.parameters.spec_vs_power_operation_len_ns,
            linewidth_threshold_hz=graph.parameters.spec_vs_power_linewidth_threshold_hz,
            max_amplitude_opx=graph.parameters.spec_vs_power_max_amplitude_opx,
            min_peak_fraction=graph.parameters.spec_vs_power_min_peak_fraction,
        )
        qubit_calibration.add_node(spec_vs_power)
        # Inner loop: expand frequency/power span until a peak is found
        qubit_calibration.loop(
            spec_vs_power,
            on=should_repeat_spec_vs_power,
            max_iterations=MAX_SPEC_ITERATIONS,
        )

        # 2. Standard qubit spectroscopy – refine frequency at the calibrated power
        qubit_spec = library.nodes["03a_qubit_spectroscopy"].copy(
            name="qubit_spectroscopy",
            multiplexed=graph.parameters.multiplexed,
            frequency_span_in_mhz=graph.parameters.qubit_spec_frequency_span_mhz,
            frequency_step_in_mhz=graph.parameters.qubit_spec_frequency_step_mhz,
            operation_len_in_ns=graph.parameters.qubit_spec_operation_len_ns,
            operation_amplitude_factor=graph.parameters.qubit_spec_operation_amplitude_factor,
            num_shots=graph.parameters.qubit_spec_num_shots,
            min_peak_fraction=graph.parameters.qubit_spec_min_peak_fraction,
        )
        qubit_calibration.add_node(qubit_spec)

        # 3. Power Rabi – calibrate π-pulse amplitude (adaptive mode enabled)
        power_rabi = library.nodes["04b_power_rabi"].copy(
            name="power_rabi",
            use_adaptive=True,
            multiplexed=graph.parameters.multiplexed,
            min_amp_factor=graph.parameters.rabi_min_amp_factor,
            max_amp_factor=graph.parameters.rabi_max_amp_factor,
            amp_factor_step=graph.parameters.rabi_amp_factor_step,
            num_shots=graph.parameters.rabi_num_shots,
        )
        qubit_calibration.add_node(power_rabi)
        # Inner loop: retry Rabi only when amplitude rescaling can fix it
        # (TOO_MANY / TOO_FEW periods); NO_OSCILLATION falls through to outer loop
        qubit_calibration.loop(
            power_rabi,
            on=should_repeat_rabi_amplitude,
            max_iterations=MAX_RABI_AMP_ITERATIONS,
        )

        qubit_calibration.connect(spec_vs_power, qubit_spec)
        qubit_calibration.connect(qubit_spec, power_rabi)

    # Add the inner subgraph as a single node in the outer graph
    graph.add_node(qubit_calibration)
    # Outer loop: restart the full sequence if the subgraph failed
    # (power_rabi → NO_OSCILLATION → frequency blacklisted → new search)
    graph.loop(
        qubit_calibration,
        on=should_restart_qubit_calibration,
        max_iterations=MAX_OUTER_ITERATIONS,
    )


graph.run()
