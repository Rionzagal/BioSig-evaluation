#include "Log.h"

static const char* LogLevels_str[] =
{
	"ERROR",
	"WARNING",
	"INFO"
};

void Log::SetLevel(LogLevels level)
{
	m_LogLevel = level;
	std::cout << "[LOGGING]: Log level set to -> " << LogLevels_str[level] << std::endl;
}

void Log::DisplayError(const char* message)
{
	if (LogError <= m_LogLevel)
		std::cout << "[ERROR]: " << message << std::endl;
}

void Log::DisplayWarning(const char* message)
{
	if (LogWarning <= m_LogLevel)
		std::cout << "[WARNING]: " << message << std::endl;
}

void Log::DisplayInfo(const char* message)
{
	if (LogInfo <= m_LogLevel)
		std::cout << "[INFO]: " << message << std::endl;
}

LogLevels Log::m_LogLevel = LogLevels::LogInfo;