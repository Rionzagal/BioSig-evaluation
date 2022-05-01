from dataclasses import InitVar, dataclass, field
from typing import ClassVar
from numpy import array, ndarray, number, random, zeros
from pandas import DataFrame
from scipy.signal import find_peaks


@dataclass(frozen=True)
class NeuronType:
    """The Izhikevich neuron types provided for each neuron"""

    TONIC_SPIKING: ClassVar[dict[str ]] = {"Name" : "Tonic Spiking",                   "a" : 0.02,     "b" : 0.2,  "c" : -65,  "d" : 6     }
    PHASIC_SPIKING: ClassVar[dict] = {"Name" : "Phasic Spiking",                  "a" : 0.02,     "b" : 0.25, "c" : -65,  "d" : 6     }
    TONIC_BURSTING: ClassVar[dict] = {"Name" : "Tonic Bursting",                  "a" : 0.02,     "b" : 0.2,  "c" : -50,  "d" : 2     }
    PHASIC_BURSTING: ClassVar[dict] = {"Name" : "Phasic Bursting",                 "a" : 0.02,     "b" : 0.2,  "c" : -55,  "d" : 0.05  }
    MIXED_MODE: ClassVar[dict] = {"Name" : "Mixed Mode",                      "a" : 0.02,     "b" : 0.2,  "c" : -55,  "d" : 4     }
    SPIKE_FREQUENCY_ADAPTATION: ClassVar[dict] = {"Name" : "Spike Frequency Adaptation",      "a" : 0.01,     "b" : 0.2,  "c" : -65,  "d" : 8     }
    CLASS_ONE_EXCITABILITY: ClassVar[dict] = {"Name" : "Class One Excitability",          "a" : 0.02,     "b" : -0.1, "c" : -55,  "d" : 6     }
    CLASS_TWO_EXCITABILITY: ClassVar[dict] = {"Name" : "Class Two Excitability",          "a" : 0.2,      "b" : 0.26, "c" : -65,  "d" : 0     }
    SPIKE_LATENCY: ClassVar[dict] = {"Name" : "Spike Lateny",                    "a" : 0.02,     "b" : 0.2,  "c" : -65,  "d" : 6     }
    SUBTHRESHOLD_OSCILATION: ClassVar[dict] = {"Name" : "Subthreshold Oscilation",         "a" : 0.05,     "b" : 0.26, "c" : -60,  "d" : 0     }
    RESONATOR: ClassVar[dict] = {"Name" : "Resonator",                       "a" : 0.1,      "b" : 0.26, "c" : -60,  "d" : -1    }
    INTEGRATOR: ClassVar[dict] = {"Name" : "Integrator",                      "a" : 0.02,     "b" : -0.1, "c" : -55,  "d" : 6     }
    REBOUND_SPIKE: ClassVar[dict] = {"Name" : "Rebound Spike",                   "a" : 0.03,     "b" : 0.25, "c" : -60,  "d" : 4     }
    REBOUND_BURST: ClassVar[dict] = {"Name" : "Rebound Burst",                   "a" : 0.03,     "b" : 0.25, "c" : -52,  "d" : 0     }
    THRESHOLD_VARIABILITY: ClassVar[dict] = {"Name" : "Threshold Variability",           "a" : 0.03,     "b" : 0.25, "c" : -60,  "d" : 5     }
    BISTABILITY: ClassVar[dict] = {"Name" : "Bistability",                     "a" : 0.1,      "b" : 0.26, "c" : -60,  "d" : 0     }
    DAP: ClassVar[dict] = {"Name" : "DAP",                             "a" : 1,        "b" : 0.2,  "c" : -60,  "d" : -21   }
    ACCOMODATION: ClassVar[dict] = {"Name" : "Accomodation",                    "a" : 0.02,     "b" : 1,    "c" : -55,  "d" : 4     }
    INHIBITION_INDUCED_SPIKING: ClassVar[dict] = {"Name" : "Ihibition Induced Spiking",       "a" : -0.02,    "b" : -1,   "c" : -60,  "d" : 8     }
    INHIBITION_INDUCED_BURSTING: ClassVar[dict] = {"Name" : "Inhibition Induced Bursting",     "a" : -0.026,   "b" : -1,   "c" : -45,  "d" : -2    }

    @classmethod
    def get_by_name(cls, name : str) -> list :
        """Retrieve the different values of the neuron types as a list of dictionaries with a common given name."""

        result : list[dict] = list()
        cls_data = cls.__dict__
        for data in cls_data.values() :
            try :
              if name in data["Name"] :
                result.append(data)
            except (TypeError, KeyError) :
              continue
        return result



@dataclass(kw_only=True) 
class Neuron :
    """
        Izhikevich neuron model generation.
        v0 [mV]: Determines the initial standby voltage value.
        type [string]: Determines the neuron type, which will determine its voltage signal behaviour.
            
        is_excitatory [bool]: Determines if the output current value of the neuron will be positive or negative.
            True: The neuron is excitatory. Therefore, the output current value will be positive.
            False: The neuron is inhibitory. Therefore, the output current value will be negative.
        _tau [unsigned float]: Determines the time-step value for each iteration the evaluation of the neuron.
        """
    ## CLASS VARIABLES
    _tau : ClassVar[float] = 0.025

    ## INIT ONLY VARIABLES
    __type : InitVar[dict | str] = field(NeuronType.TONIC_SPIKING)

    ## INSTANCE VARIABLES
    v0 : float | int = field(default=-60.)
    is_excitatory : bool = field(default=True)
    average_white_noise : float = field(default=1., repr=False)
    _type_name : str = field(init=False, default="")
    _values : dict = field(init=False, repr=False)

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

    ## INSTANCE METHODS
    def set_type(self, input_type : dict) -> None :
        """Set the neuron type as the input_type recovered from the NeuronTypes class and retrieve the values the values' database"""

        MANDATORY_KEYS = {"Name", "a", "b", "c", "d"}   #A set of mandatory keys that must be present in the input_type dictionary
        for key, value in  input_type.items(): 
            if key not in MANDATORY_KEYS : #Check for invalid keys in the custom input
                raise ValueError("The custom input type must only contain the keys {'Name', 'a', 'b', 'c', 'd'} with its respective values.")
            if ("Name" == key and not isinstance(value, str)) or (not isinstance(value, (float, int))): #Check for invalid values in the custom input
                raise TypeError("The value of 'Name' must be of type 'str', and the values of 'a', 'b', 'c', 'd' must be real numbers.")

        self._type = input_type["Name"]     #Set the type name once it has been validated
        self._values = { key : input_type[key] for key in input_type if key != "Name" }   #Set the parameter dictionary once it has been validated

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

        V += Neuron._tau * (0.04 * (V ** 2) + 5 * V + 140 - u + I_in)
        u += Neuron._tau * self._values['a'] * (self._values['b'] * V - u)

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
        self.set_type(self.__type)



@dataclass(kw_only=True)
class Network :
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
            if isinstance(weight_matrix, (DataFrame, ndarray)) and weight_matrix.shape != (self._total_neurons, self._total_neurons):
                raise ValueError("Invalid input! The dimensions of the input weight_matrix must be square and equal to the total number of neurons of the network.")
            if (isinstance(weight_matrix, ndarray) and weight_matrix.dtype != number) or (isinstance(weight_matrix, DataFrame) and weight_matrix.select_dtypes(exclude=['number']).any()):
                raise ValueError("Invalid datatype in matrix! The matrix must only contain numbers.")
            self._weights = weight_matrix
            return self._weights

        if not labels: #check if a list of labels was provided
            labels = [f"n_{x}" for x in range(self._total_neurons)] #generate a standard list of labels if none were provided
            self._labels = labels
        elif self._total_neurons > len(labels):
            raise ValueError("The list of labels must contain elements equal or greater than the number of neurons in the network!")

        if not (0 < conn_rate < 1):
            raise ValueError("The connection rate must be a positive number between 0 and 1!")
        
        if 0 >= exc_cap or 0 >= inh_cap:
            raise ValueError("The excitation and inhibition caps must be positive numbers greater than 0!")
            
        conn_values = array([[(random.random() > (1 - conn_rate)) for _ in range(self._total_neurons)] for _ in range(self._total_neurons)])
        weight_values = zeros(conn_values.shape, dtype=float) #generate a square weight matrix
        
        for j in range(weight_values.shape[0]):
            for i in range(weight_values.shape[1]):
                if conn_values[j, i] and labels[j] != labels[i]:
                    #assign a random weight value to the (j, i)_th cell of the matrix, representing the connection weight
                    #between the j_th neuron and the i_th neuron. positive if the ith neuron is excitatory, else, negative, capped by the network
                    weight_values[j, i] = (exc_cap if self.neurons[i].is_excitatory else -inh_cap) * random.random()
        self._weights = DataFrame(weight_values, index=labels, columns=labels)
        return self._weights

    def print_weights(self):
        """Prints the weight matrix of the network to the console."""
        print(f"Weights and connection matrix:\n{self._weights.to_string()}")

    def run_in_period(self, T : int, I_in : float = 0, trigger_pos : int = 0, trigger_duration : int = 200, trigger_cap : int | float= 1, trigger_type : NeuronType = NeuronType.Tonic_Spiking) -> tuple[ndarray, DataFrame, DataFrame]:
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

        # trigger parameters and responses
        trigger_neuron : Neuron = Neuron(tau=Neuron._tau, is_excitatory=True)
        _, trigger_peaks = trigger_neuron.activate(T=trigger_duration, I_in=I_in)
        I_net[trigger_pos] = trigger_peaks.size * trigger_cap

        for _ in range(int(T / Neuron._tau)):
            v_field.append(sum(v))
            I_net = [self.excitation_input * random.randn() if n.is_excitatory else self.inhibition_input * random.randn() for n in self.neurons]
            fired = [30 <= v[idx] for idx, _ in enumerate(v)]
            I_net += [sum(self._weights.to_numpy()[idx, :] * fired) for idx in range(self._total_neurons)]
            for n_idx, label in enumerate(self._labels) :
                v_individual[label].append(v[n_idx])
                v[n_idx], u[n_idx] = self.neurons[n_idx].calculate_step(v[n_idx], u[n_idx], I_net[n_idx])
                neuron_voltage[label].append(fired[n_idx])
        return array(v_field), DataFrame(v_individual), DataFrame(neuron_voltage)

    def __post_init__(self) -> None:
        if 0. >= self.excitation_input or 0. >= self.inhibition_input :
            raise ValueError("The excitation and inhibition inputs must have positive, greater than 0 values!")
        
        if 0 < len(self.neurons) :
            self._total_neurons = len(self.neurons)
            self.set_weight_matrix(labels=self._labels)

    def __repr__(self) -> str:
        message: str = f"""New Izhikevich network has been generated!
        Total neurons: {self._total_neurons}"""
        return message
