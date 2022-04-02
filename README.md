# Izhikevich neuron model simulator
This algorithm focus on the generation of a neural network model based on the Izhikevich neural model behavior simulation. The Izhikevich neural model is based on the *Hodkin-Huxley neural model*, which involves the use of first-order ODEs in order to estimate the ion channels activation functions and generate an ionic output flow. 

This project focuses on generating a model in which the user can use the Izhikevich neurons and visualize their behavior either as a single unit or combined in a network. The simulation includes settings such as input value, neuron positions and field voltage response, as well as single response.

Different modules are considered for this project, including the single-response module, the simple-network module and the real-world simulation module.

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
- [x] Store all of the neuron classes of the Izhikevich model in a separate .csv file along with its values in order to provide the right parameters to the neuron instance depending on the type.
- [x] Generate a data class for the Izhikevich neural model, including its attributes and behaviors.
- [ ] Generate a random network algorithm to represent the connection between each of the neurons presented in a desired time period.
- [ ] Generate a visualization algorithm to represent the neural field response and the neural events for the evaluation time.
- [ ] Generate an evaluation algorithm to calculate the field response rythm as an EEG channel response.

[Hodkin & Huxley experiment]: images/HnH_experiment.PNG "Hodkin and Huxley experiment in squid ginant axon"
[HnH ODE]: images/HnH_equation.PNG "Hodking and Huxley ionic current ODE model"
[HnH voltage response]: images/HnH_result.PNG "Hodkin and Huxley voltage response"