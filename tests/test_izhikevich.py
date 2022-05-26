from pandas import DataFrame
from medsignal.eeg.izhikevich import Network, Neuron, NeuronTypes
from numpy import arange, float16, linspace, random
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
        test_type = NeuronTypes.from_dict(test_data)
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
    test_neurons = {f"{n + 1}": Neuron((-60*random.random()), is_excitatory=(0.5 > random.random())) for n in range(20)}
    test_network = Network(neurons = test_neurons)
    assert test_network.total_neurons == len(test_neurons)
    assert len(test_network) == len(test_neurons)
    return


def test_invalid_weight_setting() -> None:
    assert 0 == 1
    return


def test_setting_valid_existing_weights() -> None:
    """Test the behavior of the network when a valid weight matrix is given."""
    assert 0 == 1
    return