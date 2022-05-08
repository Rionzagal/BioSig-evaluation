![Tests](https://github.com/Rionzagal/MedSig/actions/workflows/tests.yml/badge.svg)
![Issues](https://img.shields.io/github/issues/Rionzagal/MedSig)
![License](https://img.shields.io/github/license/Rionzagal/MedSig)

# MedSig package for biological signals processing and simulation
This module focuses on the simulation and specific processing of biological signals, such as EMG, EEG, and others.

## EEG module
This module contains the necesary actions for the processing and evaluation of EEG signals contained in a `numpy` array.

### Izhikevich
This module focuses on generating a model in which the user can use the Izhikevich neurons and visualize their behavior either as a single unit or combined in a network. The simulation includes settings such as input value, neuron positions and field voltage response, as well as single response.

## Package requirements
In order to successfuly use this module, a number of dependencies are needed. These dependencies can be installed or updated via the command `pip install` from the command prompt. These required packages are listed as dependencies of this package, and will be installed automatically when this package is installed.
- Pandas
- Matplotlib
- Scipy
- Numpy

[Hodkin & Huxley experiment]: DOCS/images/HnH_experiment.PNG "Hodkin and Huxley experiment in squid ginant axon"
[HnH ODE]: DOCS/images/HnH_equation.PNG "Hodking and Huxley ionic current ODE model"
[HnH voltage response]: DOCS/images/HnH_result.PNG "Hodkin and Huxley voltage response"
