#include <iostream>
#include "Models/Log.h"

int main()
{
    Log::SetLevel(LogInfo);
    Log::DisplayInfo("Hello world!");
    std::cin.get();
}

