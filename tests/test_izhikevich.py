from pandas import DataFrame
from medsignal.eeg.izhikevich import Network, Neuron, NeuronTypes
from numpy import random
import pytest


@pytest.mark.parametrize("input_name,expected", [
    ("Tonic Spiking", NeuronTypes.TONIC_SPIKING),
    ("TONIC spiking", NeuronTypes.TONIC_SPIKING),
    ("Tonic BURSTING", NeuronTypes.TONIC_BURSTING),
    ("dap", NeuronTypes.DAP),
    ("AccOMOdaTion", NeuronTypes.ACCOMODATION),
    ("Bistability", NeuronTypes.BISTABILITY)
])
def test_recover_single_neuron_type(input_name, expected) -> None:
    """Test the NeuronTypes.get_single() method with various names and result values."""
    assert NeuronTypes.get_single(input_name) is expected
    return


@pytest.mark.parametrize("input_name,expected_length", [
    ("tonic", 2),
    ("spik", 6),
    ("burst", 4),
    ("bistability", 1)
])
def test_recover_multiple_types_by_name(input_name, expected_length) -> None:
    """
    Test the NeuronTypes.get_by_name() method with different keywords and the expected lengths of 
    the resulting lists.
    """
    resulting_types: list[dict[str, str | float]] = NeuronTypes.get_by_name(input_name)
    assert len(resulting_types) == expected_length
    return


@pytest.mark.parametrize("test_type,test_activity,test_v0", [
    (NeuronTypes.TONIC_SPIKING, True, -65),
    (NeuronTypes.TONIC_BURSTING, False, -48),
    (NeuronTypes.ACCOMODATION, False, -50.2),
    (NeuronTypes.DAP, True, -88.5),
    (NeuronTypes.CLASS_ONE_EXCITABILITY, True, -75.2),
    (NeuronTypes.MIXED_MODE, True, -2.25),
    (NeuronTypes.INTEGRATOR, False, -66.8)
])
def test_neuron_creation(test_type, test_activity, test_v0):
    """Create a neuron model using the defined list of parameters and evaluate its behavioral attributes"""
    test_message: str = "Excitatory" if test_activity else "Inhibitory"
    test_neuron: Neuron = Neuron(_type=test_type, is_excitatory=test_activity, v0=test_v0)
    neuron_text: str = test_neuron.__repr__()
    type_name, _ = test_neuron.get_type()
    assert type_name == str(test_type["Name"])
    assert test_message in neuron_text
    assert str(test_v0) in neuron_text
    return


@pytest.mark.parametrize("test_tau,test_period,expected_steps", [
    (0.025, 1000, 40_000),
    (0.01, 500, 50_000),
    (0.1, 80, 800),
    (0.002, 200, 100_000)
])
def test_tau_attribute(test_tau, test_period, expected_steps):
    """Test the tau attribute and behavior with its functionality"""
    Neuron.set_tau(test_tau)
    assert int(test_period / Neuron._tau) == expected_steps
    return


@pytest.mark.parametrize("test_type", [
    NeuronTypes.TONIC_BURSTING,
    NeuronTypes.PHASIC_SPIKING,
    NeuronTypes.PHASIC_BURSTING,
    NeuronTypes.ACCOMODATION,
    NeuronTypes.INHIBITION_INDUCED_BURSTING,
    NeuronTypes.INHIBITION_INDUCED_SPIKING,
    NeuronTypes.RESONATOR,
    NeuronTypes.REBOUND_BURST,
    NeuronTypes.REBOUND_SPIKE
])
def test_type_setting(test_type):
    """Test the bahavior of the type setter method of the Neuron model"""
    test_neuron: Neuron = Neuron()
    test_neuron.set_neuron_type(test_type)
    type_name, _ = test_neuron.get_type()
    assert str(test_type["Name"]) == type_name
    return


@pytest.mark.parametrize("params, t_span, test_I, test_tau", [
    ((-60, True, 0.5), 500, 10, 0.025),
    ((-55, True, 1), 200, 5, 0.03),
    ((-45, False, 0.25), 1000, 6, 0.01),
    ((-26, False, 0.95), 800, 10, 0.05)
])
def test_neuron_run(params, t_span, test_I, test_tau):
    v0, exc, awn = params
    test_neuron: Neuron = Neuron(v0, exc, awn)
    Neuron.set_tau(test_tau)
    response, _ = test_neuron.activate(t_span, test_I)
    assert int(t_span / Neuron._tau) == len(response)
    return


@pytest.mark.parametrize("quantity, exc_inh, conn_params, run_params", [
    (10, (0.8, 1.22), (0.5, 1.2, 4.1), (100, 5, 1, 100, 4)),
    (15, (0.05, 1.45), (0.3, 3.4, 1.6), (500, 14, 0, 50, 2.2)),
    (45, (2.45, 0.22), (0.8, 1, 6), (200, 7, 20, 200, 1.5)),
    (100, (0.65, 4.33), (0.1, 6.4, 2), (350, 11, 81, 500, 2)),
    (5, (3, 0.5), (0.2, 5, 0.1), (1000, 0, 2, 300, 0.5))
])
def test_network(quantity, exc_inh, conn_params, run_params) -> None:
    """Test designed to test the behavior of the Network model in the Izhikevich Module"""
    test_neurons: list[Neuron] = [
        Neuron(-70 * random.rand(), 0.5 > random.rand(), 0.5 + random.rand()) for _ in range(quantity)
    ]
    exc_input, inh_input = exc_inh
    net: Network = Network(test_neurons, exc_input, inh_input)
    net.print_weights()
    assert str(quantity) in net.__repr__()
    c_rate, e_cap, i_cap = conn_params
    assert (quantity, quantity) == net.set_weights(c_rate, e_cap, i_cap).shape
    labels: list[str] = [ f"neuron: {i}" for i in range(quantity) ]
    custom_weights: DataFrame = DataFrame([
        [ random.rand() if n.is_excitatory else -random.rand() for n in net.neurons ] for n in net.neurons
        ])
    custom_weights = net.set_existing_weights(custom_weights, labels)
    assert (quantity, quantity) == custom_weights.shape
    t_span, current, tr_pos, tr_duration, tr_cap = run_params
    field_response, individual_response, activations = net.run(t_span, current, tr_pos, tr_duration, tr_cap)
    assert int(t_span / Neuron._tau) == len(field_response)
    assert (int(t_span / Neuron._tau), quantity) == individual_response.shape
    return
