from os import path
from typing import ClassVar
from matplotlib import pyplot as plt
from numpy import array, linspace, ndarray, random
from pandas import DataFrame, read_csv
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from neuron_types import NeuronType

@dataclass(kw_only=True) 
class Neuron :
    """
        Izhikevich neuron model generation.
        v0 [mV]: Determines the initial standby voltage value.
        type [string]: Determines the neuron type, which will determine its voltage signal behaviour.
            Neuron Types:
                Tonic_Spiking,
                Phasic_Spiking,
                Tonic_Bursting,
                Phasic_Bursting,
                Mixed_Mode,
                Spike_Frequency_Adaptation,
                Class_One_Excitability,
                Class_Two_Excitability,
                Spike_Latency,
                Subthreshold_Oscilation,
                Resonator,
                Integrator,
                Rebound_Spike,
                Rebound_Burst,
                Threshold_Variability,
                Bistability,
                DAP,
                Accomodation,
                Inhibition_Induced_Spiking,
                Inhibition_Induced_Bursting
        is_excitatory [bool]: Determines if the output current value of the neuron will be positive or negative.
            True: The neuron is excitatory. Therefore, the output current value will be positive.
            False: The neuron is inhibitory. Therefore, the output current value will be negative.
        _tau [unsigned float]: Determines the time-step value for each iteration the evaluation of the neuron.
        _path [string]: File path to a CSV file in the same folder as this class.
        
        !!! THIS CLASS ONLY ACCEPTS A CSV FILE FOR THE VALUES ASSIGNATION. MAKE SURE THE VALUES FILE IS REFERENCED !!!
        """
    ## CLASS VARIABLES
    _tau : ClassVar[float] = 0.025
    _path : ClassVar[str] = "./PYTHON/n_values.csv"

    ## INSTANCE VARIABLES
    v0 : float | int = field(default=-60.)
    is_excitatory : bool = field(default=True)
    average_white_noise : float = field(default=1., repr=False)
    _type : NeuronType = field(default=NeuronType.Tonic_Spiking)
    _values : dict = field(init=False, repr=False)

    ## INSTANCE METHODS
    def set_neuron_type(self, input_type : NeuronType | int) -> None :
        """Set the neuron type as the input_type recovered from the NeuronTypes class and retrieve the values the values' database"""
        if isinstance(input_type, int) :
            try:
                self._type = NeuronType(input_type)
            except ValueError :
                raise ValueError("The input type is not valid! Please input an appropiate value.")

        values_matrix : DataFrame = read_csv(Neuron._path)
        mask : bool = NeuronType(self._type).name == values_matrix['full name']
        correct_values : DataFrame = values_matrix[mask]
        values_frame = correct_values.drop(columns=['full name'])
        values_frame.reset_index(drop=True, inplace=True)
        self._values = {
            'a' : values_frame['a'][0],
            'b' : values_frame['b'][0],
            'c' : values_frame['c'][0],
            'd' : values_frame['d'][0]
        }

    def calculate_step(self, V : int | float, u : int | float, I_in : int | float) -> tuple[float, float] :
        """The calculation of a single step of evaluation in the Neuron, given a voltage, a support and a current in nano Ampers. \n
        receives:
            V [float or int] -> The current voltage value present in the neuron in milivolts \n
            u [float or int] -> The current supporting value present in the neuron of the support equation \n
            I_in [float or int] -> The input current value evaluated in nano Ampers \n
        returns:
            tuple of floats containing the next iterations of the response voltage and the supporting value with the structure (V, u) \n
            V [float] -> The next response voltage iteration of the Neuron evaluated in mV \n
            u [float] -> The next supporting value iteration for the support Izhikevich equation
            """

        if 30 <= V :
            V = self._values['c']
            u += self._values['d']

        V += Neuron._tau*(0.04*(V**2) + 5*V + 140 - u + I_in)
        u += Neuron._tau*self._values['a']*(self._values['b']*V - u)

        V = 30 if 30 <= V else V + self.average_white_noise * random.randn()
        
        return (V, u)

    def activate(self, T : float | int, I_in : float | int=0) -> tuple[ndarray, ndarray]:
        """Estimate the neural voltage response in mV and its activations 
        over a given time-period in miliseconds and an input current in nano Ampers. \n
        receives: 
            T [float or int] -> The time period to evaluate the Neural response, given as the amount of miliseconds for the evaluation \n
            I_in [float or int] -> The input current to which the Neuron will respond evaluated in nanoAmpers \n
        returns: 
            A tuple of numpy arrays structured as: (response_voltage, response_peaks) \n
            response_voltage [ndarray] -> The response voltage of the neuron in the given amount of miliseconds T \n
            response_peaks [ndarray] -> The amount of activations the neuron registered in the given amount of miliseconds T
            """

        vv : list[float] = list()
        v : float = self.v0
        u : float = self.v0*self._values['b']

        for _ in range(int(T / Neuron._tau)) :
            vv.append(v)
            v, u = self.calculate_step(v, u, I_in + random.random())
        
        peaks, _ = find_peaks(vv, height=20)
        return array(vv), peaks*Neuron._tau

    def __repr__(self) -> str:
        excitatory_message : str = "Excitatory" if self.is_excitatory else "Inhibitory"
        return f"Izhikevich neuron -> Type: {self._type.name}, Activity: {excitatory_message}, Initial voltage: {self.v0} mV"

    def __post_init__(self) -> None :
        self.set_neuron_type(self._type)

    ## CLASS METHODS
    @classmethod
    def set_tau(cls, input_tau : float = 0.025) -> None : 
        """Set a new tau [in miliseconds] value for step evaluation for all of the instances of class Neuron.
        The input must be a positive float that represents the period of evaluation in miliseconds, with default in None.
        Ex. If the input is 0.025, the step evaluation each time the neuron evaluates will be every 0.025 ms, which will return
        a response array of 4000 elements when a Neuron instance invokes the activate() method for a 100 ms evaluation."""

        if not input_tau or 0. >= input_tau :
            raise ValueError("The input value is not valid! Tau must be a float greater than 0!")
        cls._tau = input_tau
    
    @classmethod
    def set_path(cls, input_path : str) -> None :
        """Set a new path for Izhikevich neural values to retrieve the neural parameters for the Izhikevich parameters"""
        if not path.isfile(input_path) or '.csv' not in input_path :
            raise ValueError(f"The input path '{input_path}' does not represents a '.csv' file! Please provide a valid path.")
        cls._path = input_path

##Main function used for testing##
def main():
    T : int = 100
    test_type : NeuronType = NeuronType.Tonic_Spiking

    neuron = Neuron(v0=-70, _type=test_type)
    print(f"New Izhikevich neuron instance created!\n{neuron}")

    try:
        Neuron.set_tau(float(input("Enter a time constant for step calculations: ")))
        print(f"The Neuron time constant has been set to: {Neuron._tau:.2f}!")
    except ValueError:
        print("The input value is not valid! We will proceed with the default value. \n")

    try: 
        T = int(input("How many miliseconds should be simulated?\nThe default value is 100 ms.\n"))
    except ValueError:
        print("The input value is not valid! We will proceed with the default value.\n")

    response, peaks = neuron.activate(T=T, I_in=14)
    simulation_time = linspace(0, T/1000, num=int(T/Neuron._tau))
    print(f"The neuron activated {len(peaks)} times in {T} miliseconds!")

    plt.figure()
    plt.plot(simulation_time, response)
    plt.title(f"single {neuron._type.name} Neuron Voltage Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [mV]")

    plt.figure()
    plt.eventplot(peaks)
    plt.title(f"single {neuron._type.name} firing response")
    plt.ylabel("Neuron")
    plt.xlabel("time [ms]")
    
    plt.show()
    
if __name__ == "__main__":
    #Testing of the main function
    main()