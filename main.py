from rfsource import Signal
from rydberg_cell import *
import numpy as np
import matplotlib.pyplot as plt

TEMP            = 293.0         # [K]
CELL_LENGTH     = 0.001         # [m]

PROBE_POWER     = 100e-9        # [W]
FWHM_PROBE      = 80e-6         # [m]
PROBE_RANGE     = (-50.0, 50.0) # Detuning range [MHz]

COUPLING_POWER  = 22e-3         # [W]
FWHM_COUPLING   = 100e-6        # [m]
DELTA_C         = 0.0           # Detuning coupling laser


def main() -> None:

    # Create Signal object
    rf_signal = Signal(
        duration = 1e-3,
        Npts = int(1e5)
    )
    # Generate simulated RF signal
    rf_signal.generate(
        'noise',
        amp = [0.2, 0.8],           # [V/m] 
        freq = [43505.5, 34090.9],  # [MHz]
        phase = [0.0, 3.0],
        freq_range = (1e5, 1e6),
        amp_range = (1e-3, 1e-2),
        complexity = 100
    )
    # Plot signal and save to file in figures
    # rf_signal.plot_signal()

    # Setup experiment
    experiment = RydbergExperiment(
        field_strength_probe = calculate_field_strength(PROBE_POWER, FWHM_PROBE),
        field_strength_coupling = calculate_field_strength(COUPLING_POWER, FWHM_COUPLING),
        coupling_detuning = DELTA_C,
        probe_detuning_range = PROBE_RANGE,
        temperature = TEMP,
        cell_length = CELL_LENGTH
    )

    # Look for frequencies in the RF signal
    results = experiment.scan_transitions(35, 45, rf_signal)
    print(f"Found {len(results)} matching transition(s).")
    

    # Plot results
    for result in results:
        fig, ax = plt.subplots()
        ax.plot(result["probe_detunings"], result["transmission"])
        ax.set_title(f"n={result['n']}")
        plt.show()

if __name__ == "__main__":
    main()