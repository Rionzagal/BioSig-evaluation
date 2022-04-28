from matplotlib.pyplot import eventplot, axes, Axes, figure, plot, show, title, xlabel, ylabel
from numpy import arange, asarray, linspace, ndarray, random, array, number, zeros
from pandas import DataFrame
from dataclasses import dataclass, field
from neuron_types import NeuronType
from izhikevich_neurons import Neuron

@dataclass(kw_only=True)
class IzhNetwork :
    """A model representing a model composed by multiple neurons with multiple weights"""

    neurons : list[Neuron] = field(default_factory=list, repr=False)
    excitation_input : int | float = field(default=1, repr=False)
    inhibition_input : int | float = field(default=1, repr=False)
    _weights : list[float] | DataFrame = field(default_factory=list, repr=False)
    _labels : list[str] | None = field(default=None, init=False, repr=False)
    _total_neurons : int = field(default=0, init=False)

    def display_neurons(self) :
        message : str = "List of neurons with its labels:\n"
        for n_instance, n_label in zip(self.neurons, self._labels) :
            message += f"{n_label}: {n_instance}\n"
        print(message)

    def set_weight_matrix(self, weight_matrix : DataFrame | ndarray | None = None, conn_rate : float = 0.6, exc_cap : int | float = 1, inh_cap : int | float = 1, labels : list[str] | None = None) -> DataFrame : 
        """Generate a weight matrix for the neural network connections and influences for each of the neuron models attached to it. 
        The resulting weight matrix will determine the relationship of each neuron with the others. \n
        Parameters:
        weight_matrix : DataFrame | None, optional, default None
            Sets the Neuron weight matrix as the matrix from this parameter; if it is None, the method generates a randomized weight matrix
            using the following parameters. The matrix must be a pandas Dataframe or a numpy array with a square shape containing only numbers 
        conn_rate : float, optional, default 0.6
            Sets a threshold for determining if a neuron is connected or not to another neuron, which will determine the weight of the
            influence towards the others. Must be positive between 0 and 1. If it leans towards 0, there will be a low connection rate,
            whereas if it leans towards 1, the connection rate will ben higher.
        exc_cap : int | float, optional, default 1
            Sets a limit for excitatory weights in excitatory connected neurons. Must be positive.
        inh_cap : int | float, optional, default 1
            Sets a limit for inhibitory weights in inhibitory connected neurons. Must be positive.
        labels : list[str] | None, optional, default None
            Sets the labels for the neurons contained in the Network, represented in the weight matrix. Must be a list containing only strings
            with a number of elements equal to the total number of neurons.

        Returns:
        A pandas DataFrame with a square data matrix representing the input weights present in the network.
        """

        if weight_matrix : #check if there is an input weight matrix
            if isinstance(weight_matrix, (DataFrame, ndarray)) and weight_matrix.shape != (self._total_neurons, self._total_neurons) :
                raise ValueError("Invalid input! The dimensions of the input weight_matrix must be square and equal to the total number of neurons of the network.")
            if (isinstance(weight_matrix, ndarray) and weight_matrix.dtype != number) or (isinstance(weight_matrix, DataFrame) and weight_matrix.select_dtypes(exclude=['number']).any()) :
                raise ValueError("Invalid datatype in matrix! The matrix must only contain numbers.")
            self._weights = weight_matrix
            return self._weights

        if not labels : #check if a list of labels was provided
            labels = [f"n_{x}" for x in range(self._total_neurons)] #generate a standard list of labels if none were provided
            self._labels = labels
        elif self._total_neurons > len(labels) :
            raise ValueError("The list of labels must contain elements equal or greater than the number of neurons in the network!")

        if not (0 < conn_rate < 1) :
            raise ValueError("The connection rate must be a positive number between 0 and 1!")
        
        if 0 >= exc_cap or 0 >= inh_cap :
            raise ValueError("The excitation and inhibition caps must be positive numbers greater than 0!")
            
        conn_values = array([[(random.random() > (1 - conn_rate)) for _ in range(self._total_neurons)] for _ in range(self._total_neurons)])
        weight_values = zeros(conn_values.shape, dtype=float) #generate a square weight matrix
        
        for j in range(weight_values.shape[0]) :
            for i in range(weight_values.shape[1]) :
                if conn_values[j, i] and labels[j] != labels[i] :
                    #assign a random weight value to the (j, i)_th cell of the matrix, representing the connection weight
                    #between the j_th neuron and the i_th neuron. positive if the ith neuron is excitatory, else, negative, capped by the network
                    weight_values[j, i] = (exc_cap if self.neurons[i].is_excitatory else -inh_cap) * random.random()
        self._weights = DataFrame(weight_values, index=labels, columns=labels)
        return self._weights

    def print_weights(self) :
        """Prints the weight matrix of the network to the console."""
        print(f"Weights and connection matrix:\n{self._weights.to_string()}")

    def run_in_period(self, T : int, I_in : float = 0, trigger_pos : int = 0, trigger_duration : int = 200, trigger_cap : int | float= 1, trigger_type : NeuronType = NeuronType.Tonic_Spiking) -> tuple[ndarray, DataFrame, DataFrame] :
        """
        Activates the network for a given amount of time evaluated in miliseconds using a trigger neuron response voltage
        
        Parameters:
        T : int 
            The period of network response evaluation time represented in miliseconds. Must be positive and greater than 0.
        I_in : float, optional, default 0
            The input current applied to the trigger neuron generated for the network.
        trigger_pos : int, optional, default 0
            The neuron index in the network to which the trigger response is applied. Must be within the boundaries of the neurons list in the network"""

        #initial values and parameters
        I_net : list[float] = [0. for x in range(self._total_neurons)]
        v = array([n.v0 for n in self.neurons]) #initial values of 'v'
        u = array([n.v0 * n._values['b'] for n in self.neurons]) #initial values of 'u'
        
        #response values
        neuron_voltage : dict[str, list[int]] = {n_label: list() for n_label in self._labels}
        v_individual : dict[str, list[float]] = {n_label : list() for n_label in self._labels}
        v_field : list[float] = list() #field voltage response (sum of all neuron voltages)

        #trigger parameters and responses
        trigger_neuron : Neuron = Neuron(tau=Neuron._tau, is_excitatory=True)
        _, trigger_peaks = trigger_neuron.activate(T=trigger_duration, I_in=I_in)
        I_net[trigger_pos] = trigger_peaks.size * trigger_cap

        for _ in range(int(T / Neuron._tau)) :
            v_field.append(sum(v))
            I_net = [self.excitation_input * random.randn() if n.is_excitatory else self.inhibition_input * random.randn() for n in self.neurons]
            fired = [30 <= v[idx] for idx, _ in enumerate(v)]
            I_net += [sum(self._weights.to_numpy()[idx, :] * fired) for idx in range(self._total_neurons)]
            for n_idx, label in enumerate(self._labels) :
                v_individual[label].append(v[n_idx])
                v[n_idx], u[n_idx] = self.neurons[n_idx].calculate_step(v[n_idx], u[n_idx], I_net[n_idx])
                neuron_voltage[label].append(fired[n_idx])
        return array(v_field), DataFrame(v_individual), DataFrame(neuron_voltage)

    def __post_init__(self) -> None :
        if 0. >= self.excitation_input or 0. >= self.inhibition_input :
            raise ValueError("The excitation and inhibition inputs must have positive, greater than 0 values!")
        
        if 0 < len(self.neurons) :
            self._total_neurons = len(self.neurons)
            self.set_weight_matrix(labels=self._labels)
    
    def __repr__(self) -> str:
        message : str = f"""New Izhikevich network has been generated!
        Total neurons: {self._total_neurons}"""
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

    network.display_neurons()
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