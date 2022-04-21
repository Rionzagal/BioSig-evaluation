#include <iostream>
#pragma once

const enum LogLevels// The levels contained in the log functionality
{
	LogError, // Set this if you only want to display the ERROR messages.
	LogWarning, // Set this if you want to display the ERROR and WARNING messages.
	LogInfo, // Set this if you want to display ERROR, WARNING and INFO messages.
};

/*
	Log model generated for message, warning and error loggings.
*/
class Log
{
private:
	// The Level that determines the inclusion or exclusion of certain messages, such as errors, warnings and traces.
	static LogLevels m_LogLevel;
public:
	// Set the global log level for message display
	static void SetLevel(LogLevels level);
	// Display an Error message to the console with a specific message
	static void DisplayError(const char* message);
	//Display a Warning message to the console with a specific message
	static void DisplayWarning(const char* message);
	//Display an Info message to the console with a specific message
	static void DisplayInfo(const char* message);
};
