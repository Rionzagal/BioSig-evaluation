from pandas import DataFrame
from medsig.eeg.izhikevich import Network, Neuron, NeuronTypes
from numpy import float16, linspace, random
import pytest

# Neuron types tests


@pytest.mark.parametrize(["test_name", "expected_result"],
                         [
    ("Tonic Spiking", NeuronTypes("Tonic Spiking", 0.02, 0.2, -65, 6)),
    ("Phasic Spiking", NeuronTypes("Phasic Spiking", 0.02, 0.25, -65, 6)),
    ("Tonic Bursting", NeuronTypes("Tonic Bursting", 0.02, 0.2, -50, 2)),
    ("Phasic Bursting", NeuronTypes("Phasic Bursting", 0.02, 0.2, -55, 0.05)),
    ("Mixed Mode", NeuronTypes("Mixed Mode", 0.02, 0.2, -55, 4)),
    ("Spike Frequency Adaptation", NeuronTypes("Spike Frequency Adaptation", 0.01, 0.2, -65, 8)),
    ("Class One Excitability", NeuronTypes("Class One Excitability", 0.02, -0.1, -55, 6)),
    ("Class Two Excitability", NeuronTypes("Class Two Excitability", 0.2, 0.26, -65, 0)),
    ("Spike Lateny", NeuronTypes("Spike Lateny", 0.02, 0.2, -65, 6)),
    ("Subthreshold Oscilation", NeuronTypes("Subthreshold Oscilation", 0.05, 0.26, -60, 0)),
    ("Resonator", NeuronTypes("Resonator", 0.1, 0.26, -60, -1)),
    ("Integrator", NeuronTypes("Integrator", 0.02, -0.1, -55, 6)),
    ("Rebound Spike", NeuronTypes("Rebound Spike", 0.03, 0.25, -60, 4)),
    ("Rebound Burst", NeuronTypes("Rebound Burst", 0.03, 0.25, -52, 0)),
    ("Threshold Variability", NeuronTypes("Threshold Variability", 0.03, 0.25, -60, 5)),
    ("Bistability", NeuronTypes("Bistability", 0.1, 0.26, -60, 0)),
    ("DAP", NeuronTypes("DAP", 1, 0.2, -60, -21)),
    ("Accomodation", NeuronTypes("Accomodation", 0.02, 1, -55, 4)),
    ("Ihibition Induced Spiking", NeuronTypes("Ihibition Induced Spiking", -0.02, -1, -60, 8)),
    ("Inhibition Induced Bursting", NeuronTypes("Inhibition Induced Bursting", -0.026, -1, -45, -2))
])
def test_recover_single_neuron_type(test_name, expected_result) -> None:
    """Test the NeuronTypes.get_single() method with various names and result values."""
    test_neuron_type = NeuronTypes.get_single(test_name)
    assert (test_neuron_type == expected_result) and (test_neuron_type is not expected_result)
    return


def test_recover_not_found_neuron_type() -> None:
    """Test the NeuronTypes.get_single() method with an invalid name."""
    test_neuron_type = NeuronTypes.get_single("hello world!")
    assert not test_neuron_type and test_neuron_type is not None
    return


@pytest.mark.parametrize(["input_name", "expected_length"],
                         [
    ("Spik", 6),
    ("Burst", 4),
    ("type", 0),
    ("", 20)
])
def test_recover_multiple_types_by_name(input_name, expected_length) -> None:
    """
    Test the NeuronTypes.get_by_name() method with different keywords and the expected lengths of
    the resulting lists.
    """
    test_results = NeuronTypes.get_range(input_name)
    assert len(test_results) == expected_length
    return


def test_generate_custom_type() -> None:
    """Test the NeuronTypes() constructor using custom values"""
    test_type = NeuronTypes("My custom value", 1, 2, 3, 4)
    assert bool(test_type)
    return


def test_generate_custom_type_from_dict() -> None:
    """Test the NeuronTypes.from_dict() method."""
    type_data = {"Name": "my custom type", "a": 1, "b": 2, "c": 3, "d": 4}
    test_type = NeuronTypes.from_dict(type_data)
    assert type_data == test_type.as_dict()
    return


def test_invalid_type_from_dict() -> None:
    """Test the error raising of the NeuronTypes.from_dict() method."""
    test_data = {"Hello": 46, "a": 1, "b": 2, "c": 4, "d": 5}
    with pytest.raises(ValueError):
        _ = NeuronTypes.from_dict(test_data)
    return


def test_neuron_types_representations() -> None:
    """Test the neuron types representations"""
    test_type_1 = NeuronTypes.get_single("tonic spiking")
    test_type_2 = NeuronTypes.get_single("tonic bursting")
    assert test_type_1 != test_type_2
    assert isinstance(test_type_1.__repr__(), str) and str(test_type_2)
    return

# Neuron tests


def test_neuron_creation():
    """Create a neuron model using the defined list of parameters and evaluate its behavioral attributes"""
    test_neuron = Neuron(-65, NeuronTypes.get_single("tonic spiking"), True)
    assert test_neuron.v0 == -65 and test_neuron.is_excitatory
    assert test_neuron.neuron_type.Name in test_neuron.__repr__()
    assert test_neuron.neuron_type.Name in str(test_neuron)
    assert NeuronTypes.get_single("tonic spiking").as_dict() == test_neuron.as_dict()["Type"]
    return


def test_tau_attribute():
    """Test the tau attribute and behavior with its functionality"""
    Neuron.set_tau(0.01)
    assert Neuron.tau() == 0.01
    return


@pytest.mark.parametrize("test_tau", [1, -1, 0])
def test_invalid_tau(test_tau) -> None:
    """Test the error raising when an invalid tau attribute is set in the neruon."""
    with pytest.raises(ValueError):
        Neuron.set_tau(test_tau)
    return


def test_neuron_equality() -> None:
    """Test the equality between two equal neurons allocated in different memory slots."""
    test_neuron_1 = Neuron()
    test_neuron_2 = Neuron()
    assert test_neuron_1 == test_neuron_2
    assert test_neuron_1 is not test_neuron_2
    return


def test_neuron_inequality() -> None:
    """Test the inequality between two different neurons."""
    test_neuron_1 = Neuron(-60, NeuronTypes.get_single("tonic spiking"))
    test_neuron_2 = Neuron(-55, NeuronTypes.get_single("dap"))
    assert test_neuron_1 != test_neuron_2
    return


@pytest.mark.parametrize(["tspan", "tau", "expected_length"],
                         [
    (200, 0.01, int(200 / 0.01)),
    (500, 0.025, int(500 / 0.025)),
    (1000, 0.5, int(1000 / 0.5))
])
def test_neuron_run(tspan, tau, expected_length) -> None:
    """Test the running method of a single neuron unit."""
    test_neuron = Neuron(-60, is_excitatory=True)
    Neuron.set_tau(tau)
    response, activations = test_neuron.activate(tspan, 14)
    assert len(response) == expected_length
    if activations.any():
        t_period = linspace(start=tau, stop=tspan, num=int(tspan / tau), dtype=float16)
        for peak in activations:
            assert peak in t_period
    return


# Network tests
def test_network_creation() -> None:
    """Test the creation of a Network type object based on a list of random neurons."""
    test_neurons = {f"{n + 1}": Neuron((-60*random.random()), is_excitatory=(0.5 > random.random())) for n in range(20)}
    test_network = Network(neurons=test_neurons)
    assert test_network.total_neurons == len(test_neurons)
    assert len(test_network) == len(test_neurons)
    for neuron in test_neurons:
        assert neuron in test_network.neurons
    assert "Total neurons = 20" in test_network.__repr__()
    assert "Izhikevich network with 20" in str(test_network)
    return


def test_network_creation_empty() -> None:
    """Test the constructor of a Network object without any input."""
    test_network = Network()
    assert 1 == test_network.thalamic_ex
    assert 1 == test_network.thalamic_in
    return


@pytest.mark.parametrize("qty", [5, 3, 9, 13, 20, 10])
def test_net_constructor_collection_unlabeled(qty) -> None:
    """Test the constructor of a Network object using a list of neurons as parameter without labels."""
    test_neurons = [Neuron((-n), random.choice(NeuronTypes.get_range()), (0.5 > random.random())) for n in range(qty)]
    net = Network(test_neurons)
    assert qty == net.total_neurons
    assert qty == len(net.neurons.keys())
    return


@pytest.mark.parametrize("neuron_qty", [5, 13, 15, 29, 40])
def test_network_with_separate_neurons(neuron_qty) -> None:
    """Test the network constructor without neurons and adding them separately."""
    test_network = Network()
    test_neurons = dict()
    for n in range(neuron_qty):
        test_neurons[f"n_{n}"] = Neuron()
    test_network.neurons = test_neurons
    assert neuron_qty == test_network.total_neurons
    return


@pytest.mark.parametrize("neuron_qty", [3, 5, 8, 16, 29])
def test_network_with_labeled_neurons(neuron_qty) -> None:
    """Test the network constructor with a list of neurons and a valid set of labels"""
    input_neurons = [Neuron(-75*random.random(), random.choice(NeuronTypes.get_range()), (0.5 > random.random())) for _ in range(neuron_qty)]
    input_labels = {f"n_{n}" for n in range(neuron_qty)}
    net = Network(input_neurons, input_labels)
    assert net.total_neurons == neuron_qty
    for label in input_labels:
        assert label in set(net.neurons.keys())
    return


def test_network_invalid_labeled_neurons() -> None:
    """Test the reaction of the network constructor when an invalid set of labels is presented with a list of neurons."""
    input_neurons = [Neuron(-75*random.random(), random.choice(NeuronTypes.get_range()), (0.5 > random.random())) for _ in range(5)]
    invalid_labels_shorter = {f"n_{n}" for n in range(4)}
    invalid_labels_longer = {f"n_{n}" for n in range(7)}
    invalid_labels_repeated = ["0", "1", "1", "5", "6"]
    with pytest.raises(ValueError):
        _ = Network(input_neurons, invalid_labels_longer)
    with pytest.raises(ValueError):
        _ = Network(input_neurons, invalid_labels_shorter)
    with pytest.raises(ValueError):
        _ = Network(input_neurons, invalid_labels_repeated)
    return


def test_invalid_neurons_for_network() -> None:
    """Test the reaction when an invalid object type is set to Network.neurons."""
    test_network = Network()
    with pytest.raises(TypeError):
        test_network.neurons = [Neuron() for n in range(5)]
        test_network.neurons = Neuron(-50)
        test_network.neurons = {Neuron(-10*n) for n in range(5)}
    return


@pytest.mark.parametrize("input", [-0.2, 0, "2.1"])
def test_invalid_thalamic_ex(input) -> None:
    """Test the reaction of the thalamic_ex setter when an invalid input is presented."""
    net = Network()
    if isinstance(input, (int | float)):
        with pytest.raises(ValueError):
            net.thalamic_ex = input
    else:
        with pytest.raises(TypeError):
            net.thalamic_ex = input
    return


@pytest.mark.parametrize("input", [-0.2, 0, "2.1"])
def test_invalid_thalamic_in(input) -> None:
    """Test the reaction of the thalamic_in setter when an invalid input is presented."""
    net = Network()
    if isinstance(input, (int | float)):
        with pytest.raises(ValueError):
            net.thalamic_in = input
    else:
        with pytest.raises(TypeError):
            net.thalamic_in = input
    return


def test_invalid_weight_setting() -> None:
    """Test the reaction of the weiths setter when an invalid weigths matrix is given"""
    net = Network([Neuron() for n in range(2)])
    inv_wm_te = [[2, 1], [1, 2]]
    inv_wm_Nan = DataFrame([["1", 1], [2, "j"]])
    inv_wm_sh = DataFrame([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(TypeError):
        net.weights = inv_wm_te
    with pytest.raises(ValueError):
        net.weights = inv_wm_Nan
    with pytest.raises(ValueError):
        net.weights = inv_wm_sh
    return


@pytest.mark.parametrize("qty", [2, 4, 8, 13, 16, 20, 31])
def test_setting_valid_existing_weights(qty) -> None:
    """Test the behavior of the network when a valid weight matrix is given."""
    net = Network([Neuron() for n in range(qty)])
    valid_weights = DataFrame(random.rand(qty, qty))
    net.weights = valid_weights
    assert (net.total_neurons, net.total_neurons) == valid_weights.shape
    return
