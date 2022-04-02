# Izhikevich neuron model simulator

This algorithm focus on the generation of a neural network model based on the Izhikevich neural model behavior simulation. The Izhikevich neural model is based on the *Hodkin-Huxley neural model*, which involves the use of first-order ODEs in order to estimate the ion channels activation functions and generate an ionic output flow. 

![Hodkin & Huxley voltage observation from a squid's giant axon][Hodkin & Huxley experiment]

## Development


The ionic current model obtained from their investigation is presented as follows.
![Hodkin & Huxley Current response equation][HnH ODE]

![Hodkin & Huxley voltage response obtained from the ODE model][HnH voltage response]

## Package requirements
In order to successfuly use this module, a number of dependencies are needed. These dependencies can be installed or updated via the command `pip install` from the command prompt.

- Pandas
- Matplotlib
- Scipy
- Numpy

# Project tasks
- [x] Store all of the neuron classes of the Izhikevich model in order to provide the right parameters.
- [ ] Generate an object class for the Izhikevich neural model and its activities.
- [ ] Generate a random network algorithm to represent the connection between each of the neurons presented in a desired time period.
- [ ] Generate a visualization algorithm to represent the neural field response and the neural events for the evaluation time.
- [ ] Generate an evaluation algorithm to calculate the field response rythm as an EEG channel response.

[Hodkin & Huxley experiment]: images/HnH_experiment.PNG "Hodkin and Huxley experiment in squid ginant axon"
[HnH ODE]: images/HnH_equation.PNG "Hodking and Huxley ionic current ODE model"
[HnH voltage response]: images/HnH_result.PNG "Hodkin and Huxley voltage response"