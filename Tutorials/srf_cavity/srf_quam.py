from dataclasses import field
from typing import Dict

from quam.core import quam_dataclass

from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam
from quam_builder.architecture.superconducting.cavity.cavity import Cavity

# Re-use the standard TemporaryCalibrationData so QUAM's type checker accepts
# state.json entries written by standard calibration nodes.
from quam_config.my_quam import TemporaryCalibrationData  # noqa: F401

from quam_builder.architecture.superconducting.qubit_pair import CavityTransmonPair  # noqa: F401


@quam_dataclass
class SrfQuam(FixedFrequencyQuam):
    """QUAM for a fixed-frequency transmon coupled to SRF cavities (Alice, Bob) on OPX+/Octave.

    Extends FixedFrequencyQuam with:
      - cavities: SRF storage cavities (Alice, Bob, ...)
      - cavity_transmon_pairs: qubit–cavity coupling parameters (chi, displacement k)
      - temp_calibration: per-qubit temporary state used by adaptive calibration nodes

    EF pulses (EF_x180, EF_x90, etc.) are stored on the same xy channel as GE
    pulses.  QUA programs switch frequency via update_frequency() before EF gates.

    The load() override fixes a QUAM serialisation quirk where Octave loopbacks
    (Tuple[Tuple[str,str],str]) are round-tripped through JSON as nested lists,
    which typeguard rejects on reload.
    """

    cavities: Dict[str, Cavity] = field(default_factory=dict)
    cavity_transmon_pairs: Dict[str, CavityTransmonPair] = field(default_factory=dict)
    temp_calibration: Dict[str, TemporaryCalibrationData] = None

    @classmethod
    def load(cls, filepath_or_dict=None, **kwargs) -> "SrfQuam":
        # QUAM serialises Tuple[Tuple[str,str],str] loopbacks as nested JSON
        # arrays.  typeguard rejects the inner list when validating Tuple[str,str],
        # so we fix the raw dict before handing it to the QUAM instantiator.
        if isinstance(filepath_or_dict, dict):
            contents = filepath_or_dict
        else:
            serialiser = cls.get_serialiser()
            contents, _ = serialiser.load(filepath_or_dict)

        for oct_data in contents.get("octaves", {}).values():
            if isinstance(oct_data, dict):
                oct_data["loopbacks"] = [
                    (tuple(src) if isinstance(src, list) else src, dst)
                    for src, dst in oct_data.get("loopbacks", [])
                ]

        return super().load(contents, **kwargs)
