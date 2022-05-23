from dataclasses import dataclass, field
from tkinter.ttk import LabeledScale
from typing import ClassVar, Iterable, overload
from numpy import array, float16, iterable, nonzero, random, zeros
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.signal import find_peaks
from __future__ import annotations

@dataclass(init=False, frozen=True)
class NeuronTypes:
    """The Izhikevich neuron type containing the behavior variables for the neurons.
    
    Generates an immutable object containing the behavior variables for the Izhikevich
    equations for the correct function of the neuron.

    Attributes:
        Name (str): The name tag for referring to the NeuronType object.
        a (float): The 'a' value refered on the Izhikevich equations.
        b (float): The 'b' value refered on the Izhikevich equations.
        c (float): The 'c' value refered on the Izhikevich equations.
        d (float): The 'd' value refered on the Izhikevich equations.
    """
    @property
    def Name(self) -> str:
        """The name tag for referring to the NeuronType object."""
        return self._NeuronTypes__Name
    
    @property
    def a(self) -> float:
        """The 'a' value refered on the Izhikevich equations."""
        return self._NeuronTypes__a
    
    @property
    def b(self) -> float:
        """The 'b' value refered on the Izhikevich equations."""
        return self._NeuronTypes__b
    
    @property
    def c(self) -> float:
        """The 'c' value refered on the Izhikevich equations."""
        return self._NeuronTypes__c
    
    @property
    def d(self) -> float:
        """The 'd' value refered on the Izhikevich equations."""
        return self._NeuronTypes__d

    @staticmethod
    def _NeuronTypes__constant_types() -> list[dict[str, str | float]]:
        """Return a list of dictionaries representing the constant built-in neuron types."""
        return [
            {"Name": "Tonic Spiking", "a": 0.02, "b": 0.2, "c": -65, "d": 6},
            {"Name": "Phasic Spiking", "a": 0.02, "b": 0.25, "c": -65, "d": 6},
            {"Name": "Tonic Bursting", "a": 0.02, "b": 0.2, "c": -50, "d": 2},
            {"Name": "Phasic Bursting", "a": 0.02, "b": 0.2, "c": -55, "d": 0.05},
            {"Name": "Mixed Mode", "a": 0.02, "b": 0.2, "c": -55, "d": 4},
            {"Name": "Spike Frequency Adaptation", "a": 0.01, "b": 0.2, "c": -65, "d": 8},
            {"Name": "Class One Excitability", "a": 0.02, "b": -0.1, "c": -55, "d": 6},
            {"Name": "Class Two Excitability", "a": 0.2, "b": 0.26, "c": -65, "d": 0},
            {"Name": "Spike Lateny", "a": 0.02, "b": 0.2, "c": -65, "d": 6},
            {"Name": "Subthreshold Oscilation", "a": 0.05, "b": 0.26, "c": -60, "d": 0},
            {"Name": "Resonator", "a": 0.1, "b": 0.26, "c": -60, "d": -1},
            {"Name": "Integrator", "a": 0.02, "b": -0.1, "c": -55, "d": 6},
            {"Name": "Rebound Spike", "a": 0.03, "b": 0.25, "c": -60, "d": 4},
            {"Name": "Rebound Burst", "a": 0.03, "b": 0.25, "c": -52, "d": 0},
            {"Name": "Threshold Variability", "a": 0.03, "b": 0.25, "c": -60, "d": 5},
            {"Name": "Bistability", "a": 0.1, "b": 0.26, "c": -60, "d": 0},
            {"Name": "DAP", "a": 1, "b": 0.2, "c": -60, "d": -21},
            {"Name": "Accomodation", "a": 0.02, "b": 1, "c": -55, "d": 4},
            {"Name": "Ihibition Induced Spiking", "a": -0.02, "b": -1, "c": -60, "d": 8},
            {"Name": "Inhibition Induced Bursting", "a": -0.026, "b": -1, "c": -45, "d": -2}
        ]

    @classmethod
    def from_dict(cls, data: dict[str, str | float]) -> NeuronTypes:
        """Generate a NeuronType object using a dictionary as data model.

        Generate a resulting NeuronType model for an Izhikevich Neuron based
        on a dictionary containing valid keys and data. The data dictionary
        must contain the following keys with their respective types:
            'Name': (str), 'a': (float), 'b': (float), 'c': (float), 'd': (float).
        
        Args:
            data (dict): A dictionary containing data with only the keys and values as follows:
                'Name': (str), 'a': (float), 'b': (float), 'c': (float), 'd': (float)

        Returns:
            NeuronType: The NeuronTypes object representation containing the dictionary data.
        
        Raises:
            ValueError: When the data dictionary contains keys different than specified in the
                argument description above.
            TypeError: When the data dictionary contains types different than specified in the
                argument description above.
        """
        MANDATORY_KEYS = {"Name", "a", "b", "c", "d"}
        MANDATORY_TYPES = {str, float, float, float, float}
        incorrect_key: bool = len(MANDATORY_KEYS) == len(data)
        for ref_key, data_key in zip(MANDATORY_KEYS, data.keys()):
            if incorrect_key:
                raise ValueError("The custom type must contain the following keys: {Name, a, b, c, d}.") 
            incorrect_key = not (ref_key in data.keys() and data_key in MANDATORY_KEYS)
        for data_value, ref_type in zip(data.values(), MANDATORY_TYPES):
            if not isinstance(data_value, ref_type):
                raise TypeError("The input data contains an invalid type!")
        return NeuronTypes(str(data["Name"]), float(data["a"]), float(data["b"]), float(data["c"]), float(data["d"]))

    @classmethod
    def get_range(cls, filter: str = "") -> list[NeuronTypes]:
        """Retrieve multiple built-in NeuronTypes based on their names.
        
        Retrieve a list containing multiple built-in, constant NeuronTypes objects based on common
        keywords in their names. If no argument is provided, a list containing all of the built-in
        NeuronTypes will be retrieved. If the keyword does not match any of Names in the built-in
        NeuronTypes, an empty list will be returned.
        
        Arguments:
            name(str) optional: The keyword used to filter the built-in NeuronTypes to be retrieved.
        
        Returns:
            list[NeuronTypes]: The resulting list containing the built-in NeuronTypes objects that
                contain the valid keyword."""
        constant_types = cls._NeuronTypes__constant_types()
        if filter == "":
            return [NeuronTypes.from_dict(data) for data in constant_types]
        result: list[NeuronTypes] = list()
        for data in constant_types:
            if filter.lower() in str(data["Name"]).lower():
                result.append(NeuronTypes.from_dict(data))
        return result

    @classmethod
    def get_single(cls, name: str) -> NeuronTypes:
        """Retrieve a single built-in NeuronTypes object matching the specified name.

        Retrieve a single built-in NeuronTypes object that matches the specified name given as
        an argument. If none of the built-in NeuronTypes matches the specified name, an empty
        NeuronTypes object will be retrieved.

        Arguments:
            name (str): The search keyword for the specified NeuronTypes to be recovered.
        
        Returns:
            NeuronTypes: The built-in NeuronTypes object whose name matches the name argument.
        """
        for data in cls._NeuronTypes__constant_types():
            if name.lower() == str(data["Name"]).lower():
                return NeuronTypes.from_dict(data)
        return NeuronTypes()

    def as_dict(self) -> dict[str, str | float]:
        """Return the current NeuronTypes object as a dictionary containing its attributes as data.
        
        Generate a new dictionary object containing the information of the current NeuronTypes object."""
        return {
            "Name": self._NeuronTypes__Name,
            "a": self._NeuronTypes__a,
            "b": self._NeuronTypes__b,
            "c": self._NeuronTypes__c,
            "d": self._NeuronTypes__d
            }

    def __init__(self, Name: str | None = None, a: float | None = None, b: float | None = None,
                 c: float | None = None, d: float | None = None) -> None:
        self._NeuronTypes__Name = Name if Name else ""
        self._NeuronTypes__a = a if a else 0.
        self._NeuronTypes__b = b if b else 0.
        self._NeuronTypes__c = c if c else 0.
        self._NeuronTypes__d = d if d else 0.
        return

    def __repr__(self) -> str:
        return f"""Neuron type of name: {self.Name} with properties
        a: {self.a:2f}, b: {self.b:2f}, c: {self.c:2f}, d: {self.d:2f}
        allocated in {id(self)}"""
    
    def __str__(self) -> str:
        return f"""Neuron type: {self.Name};
        a: {self.a}, b: {self.b}, c: {self.c}, d: {self.d}"""

    def __bool__(self) -> bool:
        data_result = self._NeuronTypes__a + self._NeuronTypes__b + self._NeuronTypes__c + self._NeuronTypes__d
        return self._NeuronTypes__Name != "" and 0 != data_result


@dataclass(init=False)
class Neuron:
    """Izhikevich Neuron model for eeg-like simulation.

    The Neuron model based on the Izhikevich equations to determine the voltage response of the Neuron membrane on
    a given time.

    Attributes:
        neuron_type (NeuronTypes): Represents the behavior of the current Neuron object as a NeuronTypes object.
        v0(float | int): Represents the initial voltage value in miliVolts of the current Neuron object.
        is_excitatory(bool): Represents the activity of the current Neuron object, wether it is excitatory or inhibitory.
        tau(float) [global]: Represents the time-step in miliseconds present between each step of the response vector.
            To assign a new value for tau, use the set_tau() method.
        average_white_noise: Represents the average white noise to be added to each point of the response vector.
    """
    _Neuron__tau: ClassVar[float] = 0.025
    average_white_noise: ClassVar[float] = 1.
    _Neuron__type: NeuronTypes = field(default_factory=NeuronTypes, init=False)
    _Neuron__v0: float | int = field(default=-60., init=False)
    _Neuron__is_excitatory: bool = field(default=True, init=False)

    def __init__(self, v0: float | None = None, n_type: NeuronTypes | None = None, is_excitatory: bool | None = None) -> None:
        self.v0 = v0 if v0 else self._Neuron__v0
        self._Neuron__type = n_type if n_type else self._Neuron__type
        self._Neuron__is_excitatory = is_excitatory if is_excitatory else self._Neuron__is_excitatory
        return

    @classmethod
    def tau(self) -> float:
        """Represents the time-step in miliseconds present between each step of the response vector."""
        return self._Neuron__tau

    @classmethod
    def set_tau(cls, value: float) -> None:
        """Assign a new global tau value for all the Neuron objects.

        Assign a new global tau value for all the Neuron objects. The given argument must be a positive number
        between 0 and 1.

        Arguments:
            value (float): Represents the newly assigned value for the global tau constant.

        Raises:
            ValueError: When the given argument is out of the interval between 0 and 1 or includes either of the
                limit values."""
        if not (0 < value < 1):
            raise ValueError("The time constant must be decimal value between 0 and 1 without including them!")
        cls._Neuron__tau = value
        return

    @property
    def v0(self) -> float:
        """Represents the initial voltage value in miliVolts of the current Neuron object."""
        return float(self._Neuron__v0)

    @v0.setter
    def v0(self, data: int | float):
        if not isinstance(data, (int, float)):
            raise TypeError(f"Invalid operation between {type(float)} and {type(data)}!")
        self._Neuron__v0 = data
        return

    @property
    def is_excitatory(self) -> bool:
        """Represents the activity of the current Neuron object, wether it is excitatory or inhibitory."""
        return self._Neuron__is_excitatory

    @property
    def neuron_type(self) -> NeuronTypes:
        """Represents the behavior of the current Neuron object as a NeuronTypes object."""
        return self._Neuron__type

    def calculate_step(self, V: int | float, u: int | float, I_in: int | float) -> tuple[float, float]:
        """Calculate the next voltage step response for the current Neuron object.

        Compute the next voltage step response for the current given Neuron object using the Izhikevich equations
        and the given data.

        Arguments:
            V(float | int): The current voltage value present in the neuron in miliVolts.
            u(float | int): The current supporting value present in the neuron of the support equation.
            I_in(float | int): The input current value evaluated in nano Ampers.

        Returns:
            tuple[float]: The next iterations of the response voltage and the supporting value with the structure (V, u).
                V(float) -> The next response voltage iteration of the Neuron evaluated in mV.
                u (float) -> The next supporting value iteration for the support Izhikevich equation."""
        if 30 <= V:
            V = float(self.neuron_type.c)
            u += float(self.neuron_type.d)
        V += Neuron.tau() * (0.04 * (V ** 2) + 5 * V + 140 - u + I_in)
        u += Neuron.tau() * float(self.neuron_type.a) * (float(self.neuron_type.b) * V - u)
        V = 30 if 30 <= V else V + self.average_white_noise * random.randn()
        return (V, u)

    def activate(self, T: float | int, I_in: float | int = 0) -> tuple[NDArray[float16], NDArray[float16]]:
        """Generate a pair of vectors with the estimated voltage response over a given period of time and an input current.

        Estimate the neural voltage response in mV and its activations over a given time-period in miliseconds and an input
        current in nano Ampers. Generate a set of responses of the computed activation of the network using the Izhikevich
        equations. Recover the response array and an activations array.

        Arguments:
            T (float | int): The time period to evaluate the Neural response, given as the amount of miliseconds for the evaluation.
            I_in (float | int): The input current to which the Neuron will respond evaluated in nanoAmpers.

        Returns:
            tuple[NDArray]: A tuple of numpy arrays structured as (response_voltage, response_peaks).
                response_voltage(ndarray): The response voltage of the neuron in the given amount of miliseconds T.
                response_peaks(ndarray): The amount of activations the neuron registered in the given amount of miliseconds T.
        """
        vv: list[float] = list()
        v: float = self.v0
        u: float = self.v0 * float(self.neuron_type.b)
        for _ in range(int(T / Neuron.tau())):
            vv.append(v)
            v, u = self.calculate_step(v, u, I_in + random.random())
        peaks, _ = find_peaks(vv, height=20)
        return array(vv, dtype=float16), array(peaks*Neuron.tau(), dtype=float16)

    def __repr__(self) -> str:
        """Return the basic parameters and behavior activity as a string."""
        excitatory_message: str = "Excitatory" if self._Neuron__is_excitatory else "Inhibitory"
        return f"""Izhikevich neuron with attributes:
        Type = {self._Neuron__type.Name},
        Activity = {excitatory_message},
        Initial voltage = {self.v0} mV
        Allocated in {id(self)}"""

    def __str__(self) -> str:
        return f"""Izhikevich neuron with type: {self._Neuron__type.Name}, {"excitatory" if self._Neuron__is_excitatory else "inhibitory"}"""

    def __post_init__(self) -> None:
        if not bool(self._Neuron__type):
            self._Neuron__type = NeuronTypes.get_single("Tonic Spiking")
        return


@dataclass(init=False)
class Network:
    """A community of Neurons evaluated together to simulate a biologic Neural Network.

    A community of Neurons evaluated together to simulate a biologic Neural Network that interacts to different
    stimuli and generates different field responses.

    Attributes:
        neurons(dict[Neuron]): The labeled neurons contained in the current Network object.
        labels(set[str]): The different unique tags used to identify the neurons in the current Network object.
        weights(DataFrame): The matrix representation of the weighted connections representing the interactions
            between each Neuron evaluated in the Network.
        thalamic_ex(float): Represents the thalamic excitation input value to the Network object response.
        thalamic_in(float): Represents the thalamic inhibition input value to the Network object response."""
    _Network__neurons: list[Neuron] = field(default_factory=list, init=False)
    _Network__weights: DataFrame = field(default=DataFrame(), init=False)
    _Network__labels: set[str] = field(default_factory=set, init=False)
    _Network__exc_inp: float = field(default=1, init=False)
    _Network__inh_inp: float = field(default=1, init=False)

    def __init__(self, neurons: list[Neuron] | None = None, weights: DataFrame | None = None,
                 labels: set[str] | list[str] | None = None, exc_inp: float | None = None, inh_inp: float | None = None) -> None:
        self._Network__neurons = neurons if neurons else self._Network__neurons
        self.weights = weights if weights else self._Network__weights
        self._Network__labels = labels if labels else self._Network__labels
        self.thalamic_ex = exc_inp if exc_inp else self._Network__exc_inp 
        self.thalamic_in = inh_inp if inh_inp else self._Network__exc_inp
        return

    @property
    def neurons(self) -> dict[str, Neuron]:
        """The labeled neurons contained in the current Network object"""
        return {n_label: neuron for n_label, neuron in zip(self._Network__labels, self._Network__neurons)}

    @neurons.setter
    def neurons(self, data: dict[str, Neuron]) -> None:
        if isinstance(data, dict):
            self._Network__neurons = list(data.values())
            self.labels = set(data.keys())
        else:
            raise TypeError("The input data must be a list of Neurons or a dictionary containing the labeled Neurons!")
        self.generate_weights()
        return

    @property
    def total_neurons(self) -> int:
        """The total number of neurons evaluated in the Network."""
        return len(self._Network__neurons)

    @property
    def labels(self) -> set[str]:
        """A set of strings representing the individual name of each neuron in the Network"""
        return self._Network__labels

    @labels.setter
    def labels(self, input_labels: set[str]) -> None:
        if len(input_labels) != self.total_neurons:
            raise ValueError("The input labels must contain the same number of labels as the total number of neurons of the network!")
        if isinstance(input_labels, set):
            self._Network__labels = input_labels
        else:
            raise TypeError("The input value is not of a valid type! The input value must be a list or a set of strings!")

    @property
    def weights(self) -> DataFrame:
        """A numerical square matrix representing the weight and connection values of the neurons in the Network."""
        return self._Network__weights

    @weights.setter
    def weights(self, weights: DataFrame | list[list[float]] | NDArray[float16]) -> None:
        if isinstance(weights, list) or weights is NDArray:
            weights = DataFrame(type=weights, index=self.labels, columns=self.labels)
        if ((weights.select_dtypes(exclude=['number'])).any()).any():
            raise ValueError("Invalid typetype in matrix! The matrix must only contain numbers.")
        if (weights.shape != (self._total_neurons, self._total_neurons)):
            raise ValueError("Invalid input! The dimensions of the input weights must be square and"
                             + "equal to the total number of neurons of the network.")
        self._Network__weights = weights
        return

    @property
    def thalamic_ex(self) -> float:
        """Represents the thalamic excitation input value to the Network object response."""
        return float(self._Network__exc_inp)

    @thalamic_ex.setter
    def thalamic_ex(self, value: float | int) -> None:
        if 0. >= value:
            raise ValueError("The excitation input value must be a positive number!")
        self._Network__exc_inp = value
        return
    
    @property
    def thalamic_in(self) -> float:
        """Represents the thalamic inhibition input value to the Network object response."""
        return float(self._Network__inh_inp)
    
    @thalamic_in.setter
    def thalamic_in(self, value: float | int) -> None:
        if 0. >= value:
            raise ValueError("The inhibition input value must be a positive number!")
        self._Network__exc_inp = value
        return

    def append_neurons(self, value: Neuron | Iterable[Neuron]) -> None:
        """Append a Neuron or a collection of Neuron objects to the Neuron objects collection in the current Network object."""
        if isinstance(value, Neuron):
            self._Network__neurons.append(value)
            tentative_label = "n_0"
            for x in self.total_neurons:
                if tentative_label in self.labels:
                    tentative_label = f"n_{x + 1}"
                else:
                    break
            self._Network__labels.add(tentative_label)
        elif isinstance(value, Iterable):
            self._Network__neurons.extend(value)
            for _ in value:
                tentative_label = "n_0"
                for x in self.total_neurons:
                    if tentative_label in self.labels:
                        tentative_label = f"n_{x + 1}"
                    else:
                        self._Network__labels.add(tentative_label)
                        break
        self.generate_weights()
        return

    def generate_weights(self, conn_rate: float = 0.6, exc_cap: int | float = 1, inh_cap: int | float = 1) -> DataFrame:
        """Generate a randomized weight matrix for the current Network object based on the given arguments.

        Generate a weight matrix for the neural network connections and influences for each of the neuron models attached to it.
        The resulting weight matrix will determine the relationship of each neuron with the others.
        
        Arguments:
            conn_rate(float) [optional]: Sets a threshold for determining if a neuron is connected or not to another neuron, which will
                determine the weight of the influence towards the others. Must be positive between 0 and 1. If it leans towards 0, there will
                be a low connection rate, whereas if it leans towards 1, the connection rate will ben higher.
            exc_cap(int | float) [optional]: Sets a limit for excitatory weights in excitatory connected neurons. Must be positive.
            inh_cap(int | float) [optional]: Sets a limit for inhibitory weights in inhibitory connected neurons. Must be positive.
            labels(list[str]) [optional]: Sets the labels for the neurons contained in the Network, represented in the weight matrix.
                Must be a list containing only strings with a number of elements equal to the total number of neurons.

        Returns:
            DataFrame: A square matrix representing the input weights present in the network."""
        if not (0 < conn_rate < 1):
            raise ValueError("The connection rate must be a positive number between 0 and 1!")
        if 0 >= exc_cap or 0 >= inh_cap:
            raise ValueError("The excitation and inhibition caps must be positive numbers greater than 0!")
        conn_type: NDArray[float16] = array([[(random.random() > (1 - conn_rate))
                                            for _ in range(self._total_neurons)]
                                            for _ in range(self._total_neurons)], dtype=float16)
        weight_type: NDArray[float16] = zeros(array(conn_type).shape, dtype=float16)
        index_labels = list(self.labels)
        for j in range(weight_type.shape[0]):
            for i in range(weight_type.shape[1]):
                if conn_type[j, i] and index_labels[j] != index_labels[i]:
                    weight_type[j, i] = (exc_cap if self._Network__neurons[i]._Neuron__is_excitatory else -inh_cap) * random.random()
        self._weights = DataFrame(weight_type, index=self.labels, columns=self.labels)
        return self._weights

    def activate(self, T: int, I_in: float = 0, trigger_pos: int = 0, trigger_duration: int = 200,
                 trigger_cap: int | float = 1) -> tuple[NDArray[float16], DataFrame, DataFrame]:
        """Activate the network for a given amount of time evaluated in miliseconds using a trigger neuron response voltage.

        Generate a set of response object representing the computed response over a given time-period presented in miliseconds.
        In order to compute the response, a trigger is generated and computed over a secondary time-period presented as a delay
        to compute the response. The trigger response is injected as an initial value to the specified neuron in the Network.

        Arguments:
            T(int): The period of network response evaluation time represented in miliseconds. Must be positive and greater than 0.
            I_in(float) [optional]: The input current applied to the trigger neuron generated for the network.
            trigger_pos(int) [optional]: The neuron index in the network to which the trigger response is applied.
                Must be within the boundaries of the neurons list in the network.
            
        Returns:
            tuple[ndarray, DataFrame, DataFrame]: The field response data of the Network activation over the given time-period
                structured as(field_voltage, individual_response, neuron_firings).
                field_voltage(ndarray): A vector containing the sum of all of the individual voltage responses over the given
                    time-period.
                individual_response(DataFrame): A DataFrame containing the individual voltage response for each of the neurons
                    evaluated in the Network over the given time-period.
                neuron_firings(DataFrame): A DataFrame containing the individual firings for each of the neurons evaluated in the
                    Network over the given time-period."""
        # Set the initial values for the run parameters.
        I_net: list[float] = [0. for _ in range(self._total_neurons)]
        v: NDArray[float16] = array([n.v0 for n in self._Network__neurons])
        u: NDArray[float16] = array([n.v0 * float(n.neuron_type.b) for n in self._Network__neurons])
        # Prepare the response type structures for the run.
        neuron_voltage: dict[str, list[int]] = {n_label: list() for n_label in self.labels}
        v_individual: dict[str, list[float]] = {n_label: list() for n_label in self.labels}
        v_field: list[float] = list()   # Prepare an empty list of respones voltage values.
        # Set a trigger neuron response run for the input current with an excitatory neuron.
        trigger_neuron: Neuron = Neuron()
        _, trigger_peaks = trigger_neuron.activate(T=trigger_duration, I_in=I_in)
        I_net[trigger_pos] = trigger_peaks.size * trigger_cap  # Assign the trigger response current to the designated neuron in the network.
        for _ in range(int(T / Neuron.tau())):
            v_field.append(sum(v))
            I_net = [
                self._Network__exc_inp * random.randn() if n.is_excitatory else self._Network__inh_inp * random.randn() for n in self._Network__neurons
                ]
            fired = [30 <= v[idx] for idx, _ in enumerate(v)]
            I_net = I_net + [sum(self.weights.to_numpy()[idx, :] * fired) for idx in range(self.total_neurons)]
            for n_idx, label in enumerate(self.labels):
                current_neuron: Neuron = self._Network__neurons[n_idx]
                v_individual[label].append(v[n_idx])
                v[n_idx], u[n_idx] = current_neuron.calculate_step(v[n_idx], u[n_idx], I_net[n_idx])
                neuron_voltage[label].append(fired[n_idx])
        return (array(v_field), DataFrame(v_individual), DataFrame(neuron_voltage))

    def __post_init__(self) -> None:
        """Generate the default randomized values of the weights and count the total number of neurons in the network."""
        if 0 < len(self.neurons):
            self._total_neurons = len(self.neurons)
            self.labels = {f"n_{x}" for x in range(self.total_neurons)}
            self.generate_weights()
            return

    def __repr__(self) -> str:
        """Present the Izhikevich network in the console with its total neurons."""
        message: str = f"""New Izhikevich network with attributes:
        Total neurons = {self.total_neurons}
        Neuron weight matrix = {self.weights.to_string()}
        Allocated at {id(self)}"""
        return message

    def __str__(self) -> str:
        return f"""Izhikevich network with {self.total_neurons} and weight matrix:
        {self.weights.to_string()}"""

    def __len__(self) -> int:
        return self.total_neurons
    
    def __add__(self, other: Network) -> Network:
        result: Network = Network()
        if isinstance(other, Network):
            result.thalamic_ex = (self.thalamic_ex + other.thalamic_ex)/2
            result.thalamic_in = (self.thalamic_in + other.thalamic_in)/2
            total_neurons = self.neurons
            for n_label, neuron in other.neurons.items():
                new_label = n_label if n_label not in total_neurons.keys() else f"{n_label}_1"
                total_neurons[new_label] = neuron
            result.neurons = total_neurons
        else:
            raise TypeError("A Network can only operate directly with another Network!")
        return result
    
    def __iadd__(self, other: Network) -> None:
        if isinstance(other, Network):
            self.thalamic_ex = (self.thalamic_ex + other.thalamic_ex)/2
            self.thalamic_in = (self.thalamic_in + other.thalamic_in)/2
            total_neurons = self.neurons
            for n_label, neuron in other.neurons.items():
                new_label = n_label if n_label not in total_neurons.keys() else f"{n_label}_1"
                total_neurons[new_label] = neuron
            self.neurons = total_neurons
        else:
            raise TypeError("A Network can only operate directly with another Network!")
        return

    def __sub__(self, other: Network) -> Network:
        result = Network()
        if isinstance(other, Network):
            result.thalamic_ex = abs(self.thalamic_ex - other.thalamic_ex)*2
            result.thalamic_in = abs(self.thalamic_in - other.thalamic_in)*2
            for n_label, i_neuron in other.neurons.items():
                if n_label in self.neurons and i_neuron == self.neurons[n_label]:
                    self.neurons.pop(n_label)
            result.neurons = self.neurons
        else:
            raise TypeError("A Network can only operate directly with another Network!")
        return result
    
    def __isub__(self, other: Network) -> None:
        if isinstance(other, Network):
            self.thalamic_ex = abs(self.thalamic_ex - other.thalamic_ex)*2
            self.thalamic_in = abs(self.thalamic_in - other.thalamic_in)*2
            for n_label, i_neuron in other.neurons.items():
                if n_label in self.neurons and i_neuron == self.neurons[n_label]:
                    self.neurons.pop(n_label)
        else:
            raise TypeError("A Network can only operate directly with another Network!")
        return