#pragma once

enum NeuronTypes
{
    TonicSpiking,
    PhasicSpiking,
    TonicBursting,
    PhasicBursting,
    MixedMode,
    SpikeFrequencyAdaptation,
    ClassOneExcitability,
    ClassTwoExcitability,
    SpikeLatency,
    SubthresholdOscilation,
    Resonator,
    Integrator,
    ReboundSpike,
    ReboundBurst,
    ThresholdVariability,
    Bistability,
    DAP,
    Accomodation,
    InhibitionInducedSpiking,
    InhibitionInducedBursting
};

class Neuron
{
public: // Public non-static variables
    float v0;
    float AvWN; //Average white noise constant determined for noise input
    bool IsExcitatory = true; //Determining if the output current value will be positive or negative
private: // Private non-static variables
    NeuronTypes m_Type = TonicSpiking; //The type to govern the behavior of the neuron instance
    float a, b, c, d;

public: // Public static variables
    static float m_tau; //Time constant determined for response step
    const char* m_Path = "n_values.csv"; //The value path that will provide the behavior values for the neuron

public:
    void CalculateStep(float& V, float& u, float Iin);
    void ActivateInPeriod(const int Tms, float Iin = 0);
};
