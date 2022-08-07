from pandas import DataFrame


def read_from_txt(filename: str,
                  rate: int = 400,
                  start_time: float = 0,
                  end_time: float = 0,
                  channels: int = 1) -> DataFrame:
    """Extract the EEG signal data from a text file.

    Read the information of a .txt file provided by the user with the given argument data and return the corresponding signal
    further analysis.

    Args:
        filename (str): The text file name provided by the user to retrieve the signal information for further analysis.
        rate (int) optional: The sampling rate evaluated in Hz used to obtain the signal found in the file.
        start_time (int) optional: The starting 'time' evaluated with the given rate to store the signal present in the file.
        end_time (int) optional: The ending 'time' evaluated with the given rate to consider the signal present in the file.
                                 If set to 0, the function will consider all of the file data starting from the given start_time.
        channels (int) optional: The number of channels to be considered in the signal storage. Each channel is presented as
                                 a one-dimensional array in the resulting object.

    Returns:
        DataFrame: A pandas DataFrame containing the stored EEG signals found in the file.

    Raises:
        Errors"""
    if filename is None:
        raise ValueError("A filename is required!")
    match filename.strip().split('.'):
        case [*_, 'txt']:
            print("hello world")
