from biosig.eeg.simulations import Neuron, NeuronType
from matplotlib import pyplot as plt
from numpy import linspace

def test_neurons():
    T : int = 100
    test_type : NeuronType = NeuronType.Tonic_Spiking

    neuron = Neuron(v0=-70, _type=test_type)
    print(f"New Izhikevich neuron instance created!\n{neuron}")

    try:
        Neuron.set_tau(float(input("Enter a time constant for step calculations: ")))
        print(f"The Neuron time constant has been set to: {Neuron._tau:.2f}!")
    except ValueError:
        print("The input value is not valid! We will proceed with the default value. \n")

    try: 
        T = int(input("How many miliseconds should be simulated?\nThe default value is 100 ms.\n"))
    except ValueError:
        print("The input value is not valid! We will proceed with the default value.\n")

    response, peaks = neuron.activate(T=T, I_in=14)
    simulation_time = linspace(0, T/1000, num=int(T/Neuron._tau))
    print(f"The neuron activated {len(peaks)} times in {T} miliseconds!")

    plt.figure()
    plt.plot(simulation_time, response)
    plt.title(f"single {neuron._type.name} Neuron Voltage Response")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [mV]")

    plt.figure()
    plt.eventplot(peaks)
    plt.title(f"single {neuron._type.name} firing response")
    plt.ylabel("Neuron")
    plt.xlabel("time [ms]")

    plt.show()