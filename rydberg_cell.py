import matplotlib.pyplot as plt
import numpy as np
import rydiqule as rq
from tabulate import tabulate
from scipy.constants import c, epsilon_0, hbar, k, physical_constants
import json
from rfsource import Signal
import arc

class RydbergExperiment():

    def __init__(self, field_strength_probe: float,
                field_strength_coupling: float,
                coupling_detuning: float,
                temperature: float,
                cell_length: float,
                probe_detuning_range: tuple[float, float] = (-30.0, 30.0),
                nmbr_detunings: int = 150,
                isotope: str = "Rb87", 
                ground_state: tuple[int, int, float] = (5,0,0.5),
                excited_state: tuple[int, int, float] = (5,1,1.5),
                ) -> None:
        """
        Initialise
        """
        # Rydberg atom isotope
        self.isotope = isotope
        self.atom = arc.Rubidium87()

        # Cell parameters
        self.temp = temperature
        self.cell_length = cell_length

        # State quantum numbers
        n_g, l_g, j_g = ground_state
        n_e, l_e, j_e = excited_state

        # Define ground state and excited state: |1> and |2>
        self.g = rq.A_QState(n_g, l_g, j_g)
        self.e = rq.A_QState(n_e, l_e, j_e)

        # Field strengths [V/m]
        self.field_strength_probe = field_strength_probe
        self.field_strength_coupling = field_strength_coupling

        # Detunings
        self.Delta_c = coupling_detuning
        self.probe_detunings = np.linspace(
            probe_detuning_range[0], probe_detuning_range[1], nmbr_detunings
        )


    def scan_transitions(self, n_min: int, n_max: int,
                          source: Signal,
                          detuning_threshold: float = 5e1,
                          amp_threshold: float = 1e-3) -> list[dict]:
        """
        Sweeps over n in [n_min, n_max) and checks whether any frequency component
        of `source` is within `detuning_threshold` MHz of the nD_{5/2} -> (n+1)P_{3/2}
        transition. For each n with at least one match, all matching RF tones are added
        to a single cell and the steady-state susceptibility is solved once.

        Args:
            n_min (int): Lower bound of n sweep (inclusive).
            n_max (int): Upper bound of n sweep (inclusive).
            source (Signal): RF signal whose frequency components are matched.
            detuning_threshold (float): Maximum allowed |f - f_trans| in MHz.
            amp_threshold (float): Minimum signal amplitude [V/m] to consider.

        Returns:
            list[dict]: One entry per matching n, each with keys:
                'n'                   : principal quantum number
                'frequencies'         : list of matched RF frequencies [MHz]
                'transition_frequency': nD -> (n+1)P transition frequency [MHz]
                'detunings'           : list of f - f_trans values [MHz]
                'probe_detunings'     : probe detuning axis [MHz]
                'chi_imag'            : Im[chi] vs probe detuning
        """
        results = []

        # Probe 5S_1/2 (mj=-0.5) -> 5P_3/2 (mj=-0.5), linear polarisation q=0
        d_probe = self.atom.getDipoleMatrixElement(5, 0, 0.5, -0.5,  5, 1, 1.5, -0.5,  0)
        scale_probe = rq.scale_dipole(abs(d_probe))
        assert abs(d_probe) > 0, "Probe dipole is zero — check mj and q"

        signal_frequencies = source.signal_params["frequencies"]
        signal_amplitudes  = source.signal_params["amplitudes"]
        signal_phases      = source.signal_params["phases"]

        for n in range(n_min, n_max + 1):
            d, p, cell = self.setup_holloway_cell(n)

            # Coupling 5P_3/2 (mj=-1.5) -> nD_5/2 (mj=-2.5), sigma- polarisation q=-1
            d_coupling = self.atom.getDipoleMatrixElement(5, 1, 1.5, -1.5,  n, 2, 2.5, -2.5, -1)

            # RF nD_5/2 (mj=-1.5) -> (n+1)P_3/2 (mj=-1.5), linear polarisation q=0
            d_RF = self.atom.getDipoleMatrixElement(n, 2, 2.5, -1.5,  n+1, 1, 1.5, -1.5,  0)

            assert abs(d_coupling) > 0, f"Coupling dipole is zero for n={n} — check mj and q"
            assert abs(d_RF)       > 0, f"RF dipole is zero for n={n} — check mj and q"

            scale_coupling = rq.scale_dipole(abs(d_coupling))
            scale_RF       = rq.scale_dipole(abs(d_RF))

            # Transition frequency |d> -> |p> in MHz (abs handles sign convention of ARC)
            trans_freq = abs(self.atom.getTransitionFrequency(n, 2, 2.5, n+1, 1, 1.5)) / 1e6

            # Add probe and coupling lasers
            cell.add_coupling(
                states = (self.g, self.e),
                rabi_frequency = self.field_strength_probe * scale_probe,
                detuning = self.probe_detunings,
                label = "probe",
                time_dependence = lambda t: 1.0
            )
            cell.add_coupling(
                states = (self.e, d),
                rabi_frequency = self.field_strength_coupling * scale_coupling,
                detuning = self.Delta_c,
                label = "coupling",
                time_dependence = lambda t: 1.0
            )

            matched_frequencies = []
            matched_detunings   = []

            for f, A, phase in zip(signal_frequencies, signal_amplitudes, signal_phases):
                Delta_RF = f - trans_freq
                if abs(Delta_RF) < detuning_threshold and A >= amp_threshold:
                    cell.add_coupling(
                        states = (p, d),
                        rabi_frequency = A * scale_RF,
                        detuning = Delta_RF,
                        phase = phase,
                        label = f"RF_{f:.4f}",
                        time_dependence = lambda t: 1.0
                    )
                    matched_frequencies.append(f)
                    matched_detunings.append(Delta_RF)

            if matched_frequencies:
                sol = rq.solve_steady_state(cell)
                chi = sol.get_susceptibility()
                transmission = sol.get_transmission_coef()
                results.append({
                    "n":                    n,
                    "frequencies":          matched_frequencies,
                    "transition_frequency": trans_freq,
                    "detunings":            matched_detunings,
                    "probe_detunings":      self.probe_detunings.copy(),
                    "chi_imag":             np.imag(chi),
                    "transmission":         transmission
                })

        return results


    def setup_holloway_cell(self, n: int, **kwargs) -> tuple[rq.atom_utils.A_QState, rq.atom_utils.A_QState, rq.cell.Cell]:
        """
        Sets up cell in Holloway configuration, with |3> = nD_{5/2} -> |4> = (n+1)P_{3/2} transition.

        Args:
            n (int): Quantum number n of the D state.
        
        Returns:
            d (rq.atom_utils.A_QState): The nD_{5/2} state.
            p (rq.atom_utils.A_QState): The (n+1)P_{3/2} state.
            cell (rq.cell.Cell): Instance of rydiqule cell with ground state, excited state and the d and p states.
        """

        d = rq.A_QState(n, 2, 2.5)      # First Rydberg state  |3> nD_{5/2}
        p = rq.A_QState(n+1, 1, 1.5)    # Second Rydberg state |4> (n+1)P_{3/2}

        return d, p, rq.Cell(
            self.isotope, 
            [self.g, self.e, d, p], 
            cell_length = self.cell_length, 
            temp = self.temp,
            **kwargs
        )

def calculate_field_strength(power: float, fwhm: float) -> float:
    """
    Calculates the field strength of a laser with power P [W] and FWHM [m] as E = sqrt(2I/c*epsilon_0) where I = P/(2pi*fwhm^2*(ln2)^2). Assumes Gaussian beam.

    Args:
        power (float): Laser power in Watts.
        fwhm (float): FWHM of the beam measured in meters.
    
    Returns:
        E (float): Field strength of the laser [V/m].
    """
    I = power / (2*np.pi*fwhm**2*np.log(2)**2)
    return np.sqrt(2.0*I / (c*epsilon_0))

def list_holloway_transitions(isotope: str, n_min: int, n_max: int) -> None:
    """
    Prints out a nicely formatted table of nD->(n+1)P transition frequencies for isotope.

    Args:
        isotope (str): Name of Rydberg isotope.
        n_min (int): Lowest n in table.
        n_max (int): Highest n in table.
    """
    data = []
    for n in range(n_min, n_max + 1):
        freq = get_holloway_transition_freq(isotope, n)
        data.append([n, freq])
    headers = ["Quantum number n", "Transition freq. [MHz]"]
    
    print(f"Transition freqs. for {isotope} in Holloway configuration")
    print(tabulate(data, headers, tablefmt="grid"))

def get_holloway_transition_freq(isotope: str, n: int) -> float:
    """
    Returns the transition frequency of nD -> (n+1)P for the given isotope.

    Args:
        isotope (str): Name of Rydberg isotope.
        n (int): Principle quantum number of D state.
    
    Returns:
        freq (float): Transition frequency [MHz].
    """
    isotope_dict = {
        "Rb87": arc.Rubidium87()
    }
    atom = isotope_dict[isotope]
    return atom.getTransitionFrequency(n+1, 1, 1.5, n, 2, 2.5) / 1e6
