from biosig.eeg.simulations import Network, Neuron, NeuronType
from numpy import random, linspace
from matplotlib import pyplot as plt

n_neurons : int = 10 #number of test neurons in the program
test_neurons : list[Neuron] = list()

for i in range(n_neurons) :
    v0 = random.randint(low=-80, high=-50)
    exc_inh = random.randn() >= 0.1
    test_neurons.append(Neuron(v0=v0, is_excitatory=exc_inh))

network : Network = Network(neurons=test_neurons)
print(network)

network.display_neurons()
network.print_weights()

T = 1000

voltage, neuron_voltage, firings = network.run_in_period(T=T, I_in=20, trigger_pos=0, trigger_duration=200, trigger_cap=1)

t_span = linspace(0, T/1000, num=int(T/Neuron._tau))

plt.figure()
plt.plot(t_span, voltage)
plt.title("Field response")
plt.xlabel("Time [seconds]")
plt.ylabel("Voltage [mV]")

neuron_firings = firings.to_numpy()

plt.figure()
plt.eventplot(neuron_firings)
plt.title("Firings")
plt.xlabel("Time [seconds]")
plt.ylabel("Neuron")

plt.show()