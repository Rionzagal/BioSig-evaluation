#include "Neuron.h"
#include "math.h"

#include <array>
#include <ctime>
#include <cstdlib>
#include <iostream>

void Neuron::CalculateStep(float& V, float& u, float Iin)
{
    std::srand(time(NULL));

    if (30.0f <= V)
    {
        V = c;
        u += d;
    }

    V += m_tau * (0.04f * (V * V) + 5.0f * V + 140.0f - u + Iin);
    u += m_tau * a * (b * V - u);

    if (30.0f <= V)
        V = 30.0f;
    else
        V += AvWN * (float)std::rand() / RAND_MAX;
}

void Neuron::ActivateInPeriod(const int Tms, float Iin = 0) 
{
    int TotalSteps = (int)std::ceil((float)Tms / m_tau);

     // TODO: Agregar maneras de generar vectores dinámicos
}
