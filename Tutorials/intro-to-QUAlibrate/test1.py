# %% {Imports}
import matplotlib.pyplot as plt

from configuration.configuration_with_lf_fem_and_mw_fem import *

from qm import QuantumMachinesManager
from qm.qua import *

from qualang_tools.results import progress_counter, fetching_tool

from qualibrate import QualibrationNode
from calibration_utils.time_of_flight import (
    Parameters,
    process_raw_data,
    fit_raw_data,
    plot_single_run_with_fit,
    plot_averaged_run_with_fit,
)
from qualibration_libs.runtime import simulate_and_p