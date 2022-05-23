from pandas import DataFrame
from medsignal.eeg.izhikevich import Network, Neuron
from numpy import random
import pytest

# Neuron types tests
def test_recover_single_neuron_type() -> None:
    """Test the NeuronTypes.get_single() method with various names and result values."""
    assert 1 == 0
    return


def test_recover_multiple_types_by_name(input_name, expected_length) -> None:
    """
    Test the NeuronTypes.get_by_name() method with different keywords and the expected lengths of
    the resulting lists.
    """
    assert 1 == 0
    return


# Neuron tests
def test_neuron_creation():
    """Create a neuron model using the defined list of parameters and evaluate its behavioral attributes"""
    assert 1 == 0
    Neuron()
    return


def test_tau_attribute():
    """Test the tau attribute and behavior with its functionality"""
    assert 1 == 0
    return


def test_invalid_tau() -> None:
    """Test the error raising when an invalid tau attribute is set in the neruon."""
    assert 1 == 0
    return


def test_valid_type_setting() -> None:
    """Test the bahavior of the type setter method of the Neuron model"""
    assert 0 == 1
    return

def test_invalid_type_setting() -> None:
    """Test the bahavior of the neuron when a neuron type is set that is not part of the NeuronTypes class."""
    assert 0 == 1
    return


def test_valid_custom_type_setting() -> None:
    """Test the behavior of the neuron when a valid custom type is set"""
    assert 0 == 1
    return


def test_invalid_custom_type() -> None:
    """Test the Error raising when an invalid custom type is set."""
    assert 0 == 1
    return


def test_neuron_run() -> None:
    assert 0 == 1
    return


# Network tests
def test_network() -> None:
    assert 0 == 1
    return


def test_invalid_weight_setting() -> None:
    assert 0 == 1
    return


def test_setting_valid_existing_weights() -> None:
    """Test the behavior of the network when a valid weight matrix is given."""
    assert 0 == 1
    return