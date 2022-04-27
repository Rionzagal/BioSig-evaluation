from cProfile import label
import numpy as np
from matplotlib.pyplot import eventplot, axes, Axes, figure, plot, show, title, xlabel, ylabel
from numpy import arange, asarray, linspace, ndarray, random, array, where
from pandas import DataFrame
from dataclasses import dataclass, field
from izhikevich_neurons import Neuron

@dataclass(kw_only=True)
class IzhNetwork(object) :
    """A model representing a model composed by multiple neurons with multiple weights"""

    global_tau : float = field(default=0.025, repr=False)
    total_neurons : int = field(default=0, init=False)
    neurons : list[Neuron] = field(default_factory=list, repr=False)
    excitation_cap : int | float = field(default=1, repr=False)
    inhibition_cap : int | float = field(default=1, repr=False)
    excitation_input : int | float = field(default=1, repr=False)
    inhibition_input : int | float = field(default=1, repr=False)
    _weights : list[float] | DataFrame = field(default_factory=list, repr=False)
    _labels : list[str] | None = field(default=None, init=False, repr=False)

    def print_neurons(self) :
        message : str = "List of neurons with its labels:\n"
        for n_instance, n_label in zip(self.neurons, self._labels) :
            message += f"{n_label}: {n_instance}\n"

        print(message)

    def generate_weight_matrix(self, conn_rate : float = 0.6, labels : list[str] | None = None) -> DataFrame : 
        if labels is None : #check if a list of labels was provided
            labels = [f"n_{x}" for x in range(self.total_neurons)] #generate a standard list of labels if none were provided
            self._labels = labels
        elif self.total_neurons > len(labels) :
            raise ValueError("The list of labels must contain elements equal or greater than the number of neurons in the network!")
            
        conn_values = array([[(random.random() > (1-conn_rate)) for x in range(self.total_neurons)] for y in range(self.total_neurons)])
        weight_values = np.zeros(conn_values.shape, dtype=np.float32) #generate a square weight matrix
        
        for j in range(weight_values.shape[0]) :
            for i in range(weight_values.shape[1]) :
                if conn_values[j, i] and labels[j] != labels[i] :
                    #assign a random weight value to the (j, i)_th cell of the matrix, representing the connection weight
                    #between the j_th neuron and the i_th neuron. positive if the ith neuron is excitatory, else, negative, capped by the network
                    weight_values[j, i] = (self.excitation_cap if self.neurons[i].is_excitatory else -self.inhibition_cap)*random.random()
        self._weights = DataFrame(weight_values, index=labels, columns=labels)

        return self._weights

    def print_weights(self) :
        print(f"Weights and connection matrix:\n{self._weights.to_string()}")

    def run_in_period(self, T : int = 1000, I_in : float = 0, trigger_pos : int = 0, trigger_duration : int = 200, trigger_cap : int | float= 1) -> tuple[ndarray, DataFrame, DataFrame]:
        #initial values and parameters
        I_net : list[float] = [0. for x in range(self.total_neurons)]
        tspan : ndarray[float] = arange(start=0, stop=T, step=self.global_tau, dtype=float)
        v = array([n.v0 for n in self.neurons]) #initial values of 'v'
        u = array([n.v0*n._values['b'][0] for n in self.neurons]) #initial values of 'u'
        
        #response values
        neuron_voltage : dict[str, list[int]] = {n_label: list() for n_label in self._labels}
        v_individual : dict[str, list[float]] = {n_label : list() for n_label in self._labels}
        v_field : list[float] = list() #field voltage response (sum of all neuron voltages)

        #trigger parameters and responses
        trigger_neuron : Neuron = Neuron(tau=self.global_tau, is_excitatory=True)
        _, trigger_peaks = trigger_neuron.activate(T=trigger_duration, I_in=I_in)
        I_net[trigger_pos] = trigger_peaks.size * trigger_cap

        for t in range(len(tspan)) :
            v_field.append(sum(v))
            I_net = [self.excitation_input*random.randn() if n.is_excitatory else self.inhibition_input*random.randn() for n in self.neurons]
            fired = [30 <= v[idx] for idx in range(len(v))]
            I_net += [sum(self._weights.to_numpy()[idx, :]*fired) for idx in range(self.total_neurons)]
            for n_idx, label in zip(range(self.total_neurons), self._labels) :
                v_individual[label].append(v[n_idx])
                v[n_idx], u[n_idx] = self.neurons[n_idx].calculate_step(v[n_idx], u[n_idx], I_net[n_idx])
                neuron_voltage[label].append(fired[n_idx])

        return array(v_field), DataFrame(v_individual), DataFrame(neuron_voltage)

    def __post_init__(self) -> None :
        if 0. >= self.global_tau :
            raise ValueError("The parameter gloabal_tau has an invalid value. Make sure that this value is greater than 0!")
        if 0. >= self.excitation_cap or 0. >= self.inhibition_cap :
            raise ValueError("The excitation and inhibition caps must have positive, greater tha 0 values!")
        if 0. >= self.excitation_input or 0. >=self.inhibition_input :
            raise ValueError("The excitation and inhibition inputs must have positive, greater than 0 values!")
        
        if 0 < len(self.neurons) :
            self.total_neurons = len(self.neurons)
            
            self.generate_weight_matrix(labels=self._labels)
    
    def __repr__(self) -> str:
        message : str = f"""New Izhikevich neuron has been generated!
        Total neurons: {self.total_neurons}; Global time-constant: {self.global_tau};
        Excitation limit: {self.excitation_cap:.2f}; Inhibition limit: {self.inhibition_cap:.2f}"""
        return message
    

def main() :
    n_neurons : int = 10 #number of test neurons in the program
    test_neurons : list[Neuron] = list()

    for i in range(n_neurons) :
        v0 = random.randint(low=-80, high=-50)
        exc_inh = random.randn() >= 0.1
        test_neurons.append(Neuron(v0=v0, is_excitatory=exc_inh))
    
    network : IzhNetwork = IzhNetwork(neurons=test_neurons)
    print(network)

    network.print_neurons()
    network.print_weights()

    T = 1000

    voltage, neuron_voltage, firings = network.run_in_period(T, 20, 0, 200, 1)

    t_span = linspace(0, T/1000, num=int(T/network.global_tau))
    
    figure()
    plot(t_span, voltage)
    title("Field response")
    xlabel("Time [seconds]")
    ylabel("Voltage [mV]")

    neuron_firings = firings.to_numpy()

    figure()
    eventplot(neuron_firings)
    title("Firings")
    xlabel("Time [seconds]")
    ylabel("Neuron")

    show()

if __name__ == "__main__" :
    main()