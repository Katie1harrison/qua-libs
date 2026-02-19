# %% {Resonator Bring-up}
"""
Resonator Bring-Up Graph

This graph finds the ideal resonator parameters (frequency and readout amplitude) through:
1. Broad resonator spectroscopy  → finds rough resonator frequency over wide span
2. Resonator spectroscopy (high power) → confirms frequency with strong signal
   → If no dip found: blacklist that frequency, loop back to step 1
3. Resonator punch-out → measures Kerr shift to find optimal readout power
4. Resonator spectroscopy (low power) → precise frequency at optimal power
"""
import logging
from typing import List, Optional

from qualibrate import (
    GraphParameters,
    QualibrationGraph,
    QualibrationLibrary,
    QualibrationNode,
)

library = QualibrationLibrary.get_active_library()

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5


class ResonatorBringUpParameters(GraphParameters):
    """Parameters for the resonator optimization graph."""
    qubits: List[str] = ["q5"]#, "q1", "q2", "q3", "q4", "q5"]

    # General
    multiplexed: bool = False

    # Broad spectroscopy
    broad_frequency_span_mhz: float = 200.0
    broad_frequency_step_mhz: float = 0.1
    broad_num_shots: int = 50
    broad_peak_prominence: float = 2
    broad_peak_width: tuple = (1, 10.0)
    blacklist_exclusion_radius_mhz: float = 10.0
    broad_readout_power_dbm: Optional[float] = 0
    broad_max_amp: float = 0.1

    # High-power spectroscopy
    high_power_frequency_span_mhz: float = 2.0
    high_power_frequency_step_mhz: float = 0.01
    high_power_num_shots: int = 100
    high_power_readout_power_dbm: Optional[float] = 0
    high_power_max_amp: float = 0.1

    # Punch-out
    punch_out_frequency_span_mhz: float = 2.0
    punch_out_frequency_step_mhz: float = 0.05
    punch_out_min_power_dbm: int = -30
    punch_out_max_power_dbm: int = 0
    punch_out_num_power_points: int = 2  # Must be 2 for shift-based analysis
    punch_out_max_amp: float = 0.1
    punch_out_num_shots: int = 100
    punch_out_frequency_shift_threshold_hz: float = 0.1e6
    use_adaptive_span: bool = True

    # Low-power spectroscopy
    low_power_frequency_span_mhz: float = 2.0
    low_power_frequency_step_mhz: float = 0.001
    low_power_num_shots: int = 100
    low_power_readout_power_dbm: Optional[float] = None
    low_power_max_amp: float = 0.1

    # Misc
    save_resonator_amplitudes: bool = True
    min_dip_contrast: float = 0.02
    lo_leakage_exclusion_mhz: float = 10.0


class ResonatorDiscoveryParameters(GraphParameters):
    """Parameters for the resonator discovery subgraph."""
    qubits: List[str] = ["q0"]
    multiplexed: bool = False


def should_retry_resonator_discovery(node, target: str) -> bool:
    """Retry the resonator_discovery subgraph when high-power spectroscopy finds no dip."""
    if node.outcomes.get(target) == "failed":
        logger.info(
            f"{target}: Resonator dip not confirmed at high power. "
            f"Retrying broad spectroscopy with blacklisted frequencies excluded."
        )
        return True
    logger.info(f"{target}: Resonator discovery succeeded.")
    return False


def should_repeat_punch_out(node: QualibrationNode, target: str) -> bool:
    """Retry the punch-out node if it failed (adaptive power span adjusts automatically)."""
    if node.outcomes.get(target) == "failed":
        logger.info(f"{target}: Punch-out failed; retrying.")
        return True
    logger.info(f"{target}: Punch-out succeeded.")
    return False


with QualibrationGraph.build(
    "resonator_optimization",
    parameters=ResonatorBringUpParameters(),
) as graph:

    # ── Inner subgraph: resonator discovery (broad → high power) ──────────────
    with QualibrationGraph.build(
        "resonator_discovery",
        parameters=ResonatorDiscoveryParameters(),
    ) as resonator_discovery:

        broad_res_spec = library.nodes["02d_broad_resonator_spectroscopy"].copy(
            name="broad_resonator_spectroscopy",
            multiplexed=graph.parameters.multiplexed,
            frequency_span_in_mhz=graph.parameters.broad_frequency_span_mhz,
            frequency_step_in_mhz=graph.parameters.broad_frequency_step_mhz,
            num_shots=graph.parameters.broad_num_shots,
            peak_prominence=graph.parameters.broad_peak_prominence,
            peak_width=graph.parameters.broad_peak_width,
            blacklist_exclusion_radius_mhz=graph.parameters.blacklist_exclusion_radius_mhz,
            readout_power_dbm=graph.parameters.broad_readout_power_dbm,
            max_amp=graph.parameters.broad_max_amp,
        )
        resonator_discovery.add_node(broad_res_spec)

        high_power_res_spec = library.nodes["02a_resonator_spectroscopy"].copy(
            name="resonator_spectroscopy_high_power",
            multiplexed=graph.parameters.multiplexed,
            frequency_span_in_mhz=graph.parameters.high_power_frequency_span_mhz,
            frequency_step_in_mhz=graph.parameters.high_power_frequency_step_mhz,
            num_shots=graph.parameters.high_power_num_shots,
            save_amplitudes=graph.parameters.save_resonator_amplitudes,
            min_dip_contrast=graph.parameters.min_dip_contrast,
            lo_leakage_exclusion_mhz=graph.parameters.lo_leakage_exclusion_mhz,
            readout_power_dbm=graph.parameters.high_power_readout_power_dbm,
            max_amp=graph.parameters.high_power_max_amp,
        )
        resonator_discovery.add_node(high_power_res_spec)

        resonator_discovery.connect(broad_res_spec, high_power_res_spec)

    graph.add_node(resonator_discovery)
    graph.loop(
        resonator_discovery,
        on=should_retry_resonator_discovery,
        max_iterations=MAX_ITERATIONS,
    )

    # ── Punch-out ──────────────────────────────────────────────────────────────
    resonator_punch_out = library.nodes["02e_resonator_punch_out"].copy(
        name="resonator_punch_out",
        multiplexed=graph.parameters.multiplexed,
        frequency_span_in_mhz=graph.parameters.punch_out_frequency_span_mhz,
        frequency_step_in_mhz=graph.parameters.punch_out_frequency_step_mhz,
        min_power_dbm=graph.parameters.punch_out_min_power_dbm,
        max_power_dbm=graph.parameters.punch_out_max_power_dbm,
        num_power_points=graph.parameters.punch_out_num_power_points,
        max_amp=graph.parameters.punch_out_max_amp,
        num_shots=graph.parameters.punch_out_num_shots,
        frequency_shift_threshold_in_hz=graph.parameters.punch_out_frequency_shift_threshold_hz,
        use_adaptive_span=graph.parameters.use_adaptive_span,
    )
    graph.add_node(resonator_punch_out)
    graph.loop(
        resonator_punch_out,
        on=should_repeat_punch_out,
        max_iterations=MAX_ITERATIONS,
    )

    # ── Low-power spectroscopy ─────────────────────────────────────────────────
    low_power_res_spec = library.nodes["02a_resonator_spectroscopy"].copy(
        name="resonator_spectroscopy_low_power",
        multiplexed=graph.parameters.multiplexed,
        frequency_span_in_mhz=graph.parameters.low_power_frequency_span_mhz,
        frequency_step_in_mhz=graph.parameters.low_power_frequency_step_mhz,
        num_shots=graph.parameters.low_power_num_shots,
        save_amplitudes=graph.parameters.save_resonator_amplitudes,
        min_dip_contrast=graph.parameters.min_dip_contrast,
        lo_leakage_exclusion_mhz=graph.parameters.lo_leakage_exclusion_mhz,
        readout_power_dbm=graph.parameters.low_power_readout_power_dbm,
        max_amp=graph.parameters.low_power_max_amp,
    )
    graph.add_node(low_power_res_spec)

    # ── Execution order ────────────────────────────────────────────────────────
    graph.connect(resonator_discovery, resonator_punch_out)
    graph.connect(resonator_punch_out, low_power_res_spec)

graph.run()
