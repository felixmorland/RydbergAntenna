import numpy as np
import matplotlib.pyplot as plt
import os
import json
import scienceplots

class Signal():
    
    def __init__(self, duration: float = 10.0, 
                 Npts: int | None = None,
                 dt: float | None = None) -> None:
        """
        Set the duration [us] of the simulated RF signal and the temporal resolution, either specifying the number of timesteps (N_pts) or the length of each timestep [us] (dt).

        /!\\ You must specify at least one of the resolution parameters!

        Args:
            duration (float): Duration of the simulated RF signal in microseconds [us].
            Npts (int, optional): Number of timesteps.
            dt (float, optional): Length of each timestep in microseconds [us].
        
        Raises:
            RuntimeError: If neither Npts or dt are specified.
        """
        self.duration = duration

        if Npts == None and dt == None:
            raise RuntimeError("/!\\ You have to specify the timesteps of the simulated RF signal! Set the value of either Npts or dt.")
        
        if Npts != None:
            self.Npts   = Npts
            self.dt     = duration / Npts
        elif dt != None:
            self.Npts   = int(duration / dt) + 1
            self.dt     = dt
        
        # Time array
        self.t = np.linspace(0, duration, self.Npts)

    def generate(self, waveform: str, 
                amp: list[float] | float = 1.0, 
                freq: list[float] | float = 1.0,
                phase: list[float] | float = 0.0,
                sig: float | None = None, 
                t0: float | None = None,
                freq_range: tuple[float, float] = (0.0, 1.0),
                amp_range: tuple[float, float] = (0.0, 1.0),
                seed: int = 2026,
                complexity: int = 10) -> None:
        """
        Generate RF signal from a set of preset waveforms;
        - constant: flat signal with constant amplitude,
        - sine: single sine wave,
        - noise: sine wave with added noise,
        - square: _ |¯| _ |¯| _,
        - sawtooth: /\\ /\\ /\\ ,
        - gaussian: Gaussian pulse with std = sig and centered at t = t0,
        - composite: sum of (complexity #) of sine waves of random frequencies and amplitudes.

        Args:
            waveform (str): Type of signal. Can either be 'constant', 'sine', 'noise', 'square', 'sawtooth', 'gaussian' or 'composite'.
            amp (float or list): Amplitude(s) [V/m] of the non-composite signals.
            freq (float or list): Frequency(ies) [MHz] of the non-composite signals.
            phase (float or list): Phase(s) [0, 2pi) of the non-composite signals.
            sig (float, optional): Std of Gaussian.
            t0 (float, optional): Where the Gaussian is centered.
            freq_range (tuple): Lowest and highest frequency in composite signal.
            amp_range (tuple): Lowest and highest amplitude in composite signal.
            seed (int): Seed for random samples.
            complexity (int): Number of sine waves in the composite signal.

        Raises:
            ValueError: If given waveform is not supported.
        """

        # Save signal parameters
        self.signal_params = {k: v for k, v in locals().items() if k != "self"}
        self.signal_params['duration'] = self.duration
        self.signal_params['Npts'] = self.Npts
        self.signal_params['dt'] = self.dt
        self.signal_params["frequencies"] = list(freq)
        self.signal_params["amplitudes"] = list(amp)
        self.signal_params["phases"] = list(phase)

        if waveform == 'constant':
            self.signal = amp * np.ones(self.Npts)

        elif waveform == 'sine':
            if type(amp) == list:
                signal = np.zeros(self.Npts)
                for f, A, phi in zip(freq, amp, phase):
                    signal += A * np.sin(2*np.pi*f*self.t + phi)
            else:
                signal = amp * np.sin(2*np.pi*freq*self.t + phase)
            self.signal = signal


        elif waveform == 'noise':
            if type(amp) == list:
                signal = np.zeros(self.Npts)
                for f, A, phi in zip(freq, amp, phase):
                    signal += A * np.sin(2*np.pi*f*self.t + phi)
            else:
                signal = amp * np.sin(2*np.pi*freq*self.t + phase)

            np.random.seed(seed)
            frequencies = np.random.uniform(low=freq_range[0], high=freq_range[1], size=complexity)
            amplitudes = np.random.uniform(low=amp_range[0], high=amp_range[1], size=complexity)
            phases = np.random.uniform(low=0.0, high=2*np.pi, size=complexity)

            for f, A, phi in zip(frequencies, amplitudes, phases):
                signal += A * np.sin(2*np.pi*f*self.t + phi)
                self.signal_params["frequencies"].append(float(f))
                self.signal_params["amplitudes"].append(float(A))
                self.signal_params["phases"].append(float(phi))
            
            self.signal = signal

        elif waveform == 'square':
            self.signal = amp * np.sign(np.sin(2*np.pi*freq*self.t))

        elif waveform == 'sawtooth':
            phase = (freq*self.t) % 1.0
            self.signal = 2.0 * amp * phase - 1.0

        elif waveform == 'gaussian':
            # Unless specified otherwise, Gaussian centered at half the signal duration
            if t0 == None:
                t0 = self.duration / 2.0
            if sig == None:
                sig = self.duration / 6.0
            
            self.signal = amp * np.exp(-0.5*((self.t - t0) / sig)**2)
        
        elif waveform == 'composite':
            np.random.seed(seed)
            frequencies = np.random.uniform(low=freq_range[0], high=freq_range[1], size=complexity)
            amplitudes = np.random.uniform(low=amp_range[0], high=amp_range[1], size=complexity)
            phases = np.random.uniform(low=0.0, high=2*np.pi, size=complexity)

            signal = np.zeros(self.Npts)
            for f, A, p in zip(frequencies, amplitudes, phases):
                signal += A * np.sin(2*np.pi*f*self.t + p)
            
            self.signal = signal
            self.signal_params["frequencies"] = [float(f) for f in frequencies]
            self.signal_params["amplitudes"] = [float(A) for A in amplitudes]
            self.signal_params["phases"] = [float(phi) for phi in phases]

        else:
            raise ValueError(f"/!\\ Unknown waveform '{waveform}'. "
                    "Choose from: constant, sine, noise, square, sawtooth, gaussian, composite")
    
    def __str__(self) -> str:
        """
        String containing the signal parameters.

        Returns:
            string (str): String containing signal parameters.
        """
        string = "Signal parameters:\n"
        for key, value in self.signal_params.items():
            string += f"{key}: {value}\n"
        return string
    
    def plot_signal(self, show: bool = False,
                    save: bool = True,
                    filename: str = 'RF_signal.pdf',
                    **kwargs) -> None:
        """
        Plots the signal in time. Saves figure as figures/filename if save = True.

        Args:
            show (bool): Whether or not to call plt.show()
            save (bool): Whether or not to save the figure.
            filename (str): Name of figure file.
            **kwargs (optional): Optional personalisations passed to ax.plot()
        """
        plt.style.use('science')
        fig, ax = plt.subplots()
        ax.plot(self.t, self.signal, **kwargs)
        ax.set_xlabel(r"Time $t\;[\mu \mathrm{s}]$")
        ax.set_ylabel(r"Field strength $E_\mathrm{RF}(t)\;[\mathrm{V/m}]$")

        if save:
            fig.savefig("figures/" + filename)
        
        if show:
            plt.show()
    
    def save_signal(self, directory: str = 'data/RF_signal') -> None:
        """
        Saves generated RF signal to signal.npy file and self.signal_params to info.json inside the directory 'directory'.

        Args:
            directory (str): Name of directory into which the signal data will be saved.
        """
        try:
            os.mkdir(directory)
            output = np.vstack((self.t, self.signal))
            np.save(directory + "/signal.npy", output)

            with open(directory + "/info.json", 'w') as outfile:
                json.dump(self.signal_params, outfile, indent=4)

        except FileExistsError:
            output = np.vstack((self.t, self.signal))
            np.save(directory + "/signal.npy", output)

            with open(directory + "/info.json", 'w') as outfile:
                json.dump(self.signal_params, outfile, indent=4)
    
    def load_signal(self, directory: str) -> None:
        """
        Loads signal found in the given directory.

        Args:
            directory (str): Name of directory where the loadable files are.
        """
        try:
            self.t, self.signal = np.load(directory + "/signal.npy")
            with open(directory + "/info.json", 'r') as infile:
                self.signal_params = json.load(infile)

        except FileNotFoundError:
            print(f"/!\\ Found no signal data in {directory} !")