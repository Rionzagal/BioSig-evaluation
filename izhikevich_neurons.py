from libs import *
from enum import Enum, auto
from numpy import arange, array, linspace, random
from pandas import DataFrame, read_csv
from scipy.signal import find_peaks
from dataclasses import dataclass, field

class NeuronType(Enum):
    """The Izhikevich neuron types provided to each neuron"""
    Tonic_Spiking = auto()
    Phasic_Spiking = auto()
    Tonic_Bursting = auto()
    Phasic_Bursting = auto()
    Mixed_Mode = auto()
    Spike_Frequency_Adaptation = auto()
    Class_One_Excitability = auto()
    Class_Two_Excitability = auto()
    Spike_Latency = auto()
    Subthreshold_Oscilation = auto()
    Resonator = auto()
    Integrator = auto()
    Rebound_Spike = auto()
    Rebound_Burst = auto()
    Threshold_Variability = auto()
    Bistability = auto()
    DAP = auto()
    Accomodation = auto()
    Inhibition_Induced_Spiking = auto()
    Inhibition_Induced_Bursting = auto()


@dataclass(kw_only=True)
class Izhikevich_neuron(object):
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
        tau [unsigned float]: Determines the time-step value for each iteration the evaluation of the neuron.
        _path [string]: File path to a CSV file in the same folder as this class.
        
        !!! THIS CLASS ONLY ACCEPTS A CSV FILE FOR THE VALUES ASSIGNATION. MAKE SURE THE VALUES FILE IS REFERENCED !!!
        """

    v0 : float | int = field(default=0.)
    I_out : float | int = field(default=0., init=False)
    is_excitatory : bool = field(default=True)
    average_white_noise : float = field(default=1., repr=False)
    tau : float = field(default=0.025, repr=False)
    _type : NeuronType = field(default=NeuronType.Tonic_Spiking)
    _path : str = field(default="./n_values.csv", repr=False)
    _values : DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None :
        if 0. > self.tau:
            raise ValueError("The tau attribute must NOT contain negative values!")

        values_matrix : DataFrame = read_csv(self._path)
        mask : bool = NeuronType(self._type).name == values_matrix['full name']
        correct_values : DataFrame = values_matrix[mask]
        self._values = correct_values.drop(columns=['full name'])
        self._values.reset_index(drop=True, inplace=True)

    def __repr__(self) -> str:
        excitatory_message : str = "Excitatory"
        if not self.is_excitatory:
            excitatory_message = "Inhibitory"
        return f"Izhikevich neuron instance. Type: {self._type.name}, Activity: {excitatory_message}, Initial voltage: {self.v0}mV"

    def activate(self, T : float | int, I_in : float | int=0) -> tuple[array, int]:
        """Estimate the voltage response in mV over a given time-period in miliseconds [T] and an input current in nano Ampers [I_in]"""
        V = self.v0
        u = self._values['b'][0]*V

        tspan : array[float] = arange(start=0, stop=T, step=self.tau, dtype=float)
        VV : list[float] = list()
        uu : list[float] = list()

        for t in tspan:
            V += self.tau*(0.04*(V**2) + 5*V + 140 - u + I_in)
            u += self.tau*self._values['a'][0]*(self._values['b'][0]*V - u)

            if 30 <= V:
                VV.append(30)
                V = self._values['c'][0]
                u += self._values['d'][0]
            else:
                VV.append(V + self.average_white_noise*random.randn())
            uu.append(u)
        
        peaks, _ = find_peaks(VV, height=20)
        return array(VV), peaks.size

##Main function used for testing##
def main():
    T : int = 100
    test_type : NeuronType = NeuronType.Tonic_Spiking

    neuron = Izhikevich_neuron(v0=-70, _type=test_type)
    print(neuron)

    try: 
        T = int(input("How many miliseconds should be simulated?\nThe default value is 100 ms.\n"))
    except ValueError:
        print("The input value is not valid! We will proceed with the default value.\n")

    response, peaks = neuron.activate(T=T, I_in=14)
    s_time = linspace(0, T/1000, num=int(T/neuron.tau))
    print(f"The neuron activated {peaks} times in {T} miliseconds!")

    plt.figure()
    plt.plot(s_time, response)
    plt.title(f"Single {neuron._type.name} Neuron Voltage Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [mV]")

    plt.show()

if __name__ == "__main__":
    #Testing of the main function
    main()