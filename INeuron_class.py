import numpy as np
import pandas as pd

def find_peaks(VV):
    pass

class Izhikevich_neuron(object):
    def __init__(self, v0, type='TS', excitatory=True, tau=0.25, values_path='./n_values.csv'):
        """
        Izhikevich neuron model generation.
        v0 [signed real number]: Determines the initial standby voltage value.
        type [string]: Determines the neuron type, which will determine its voltage signal behaviour.
            Neuron Types:
               TS: tonic spiking
               PS: phasic spiking
               TB: tonic bursting
               PB: phasic bursting
               MM: mixed mode
               FA: spike frequency adaptation
               1E: class one excitability
               2E: class two excitability
               SL: spike latency
               SO: subthresh. osc.
               R: resonator
               I: integrator
               RS: rebound spike
               RB: rebound burst
               TV: thresh. variability
               B: bistability 
               D: DAP
               A: accomodation
               IS: inhibition induced spiking
               IB: inhibition induced bursting
        excitatory [bool]: Determines if the output current value of the neuron will be positive or negative.
            True: The neuron is excitatory. Therefore, the output current value will be positive.
            False: The neuron is inhibitory. Therefore, the output current value will be negative.
        tau [unsigned float]: Determines the time-step value for each iteration the evaluation of the neuron.
        values_path [string]: File path to a CSV file in the same folder as this class.
        
        !!! THIS CLASS ONLY ACCEPTS A CSV FILE FOR THE VALUES ASSIGNATION. MAKE SURE THE VALUES FILE IS REFERENCED !!!
        """
        #Initial parameters storing
        self.v0 = v0
        self.excitatory = excitatory
        self.tau = tau
        self.I_out = 0
        #Neuron type parameters storing
        values_matrix = pd.read_csv(values_path)
        mask = type == values_matrix['abreviation']
        correct_values = values_matrix[mask]
        self.values = correct_values.drop(columns=['abreviation', 'full name'])
        self.type = correct_values['full name']

    def forward(self, T, I_in=0):
        V = self.v0
        u = self.values['b']*V
        tspan = np.arange(start=0, stop=T, step=self.tau, dtype=float)
        VV = np.array()
        uu = np.array()

        for t in tspan:
            V += self.tau*(0.04*(V**2) + 5*V + 140 - u + I_in)
            u += self.tau*self.values['a']*(self.values['b']*V - u)

            if 30 <= V:
                VV.append(30)
                V = self.values['c']
                u += self.values['d']
            else:
                VV.append(V + np.random.randn())
            uu.append(u)
        
        return VV