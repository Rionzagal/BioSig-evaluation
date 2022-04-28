from enum import Enum, auto

class NeuronType(Enum):
    """The Izhikevich neuron types provided for each neuron"""
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