import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt


def calculate_median_filter(data, window_size):
    """
    Calculates the median filter of the input data using the specified window size.

    Parameters:
    - data: NumPy array or list, the input data to calculate the median filter.
    - window_size: int, the size of the median filter window.

    Returns:
    - filtered_data: NumPy array, the filtered data using the median filter.
    """

    # Apply median filter
    filtered_data = medfilt(data, kernel_size=window_size)

    return filtered_data


def calculate_moving_average(data, window_size):
    """
    Calculates the moving average of the input data using the specified window size.

    Parameters:
    - data: NumPy array or list, the input data to calculate the moving average.
    - window_size: int, the size of the moving average window.

    Returns:
    - moving_avg: NumPy array, the moving average of the input data.
    """

    # Calculate the moving average
    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    return moving_avg


distance_data = pd.read_csv('//ad.monash.edu/home/User009/hjia0058/Desktop/CS_data/Box1_SD_1B/Distance/distances.txt', delimiter=', ', header=None)
right_dist = distance_data.iloc[:,1].to_numpy()
# back_dist = distance_data.iloc[:, 3].to_numpy()

# Process data
processed_right_dist = calculate_moving_average(right_dist, 10)
processed_right_dist = calculate_median_filter(processed_right_dist, 29)
# processed_back_dist = calculate_median_filter(back_dist, 9)

fig, axs = plt.subplots(2)

# Frame rate (Hz)
frame_rate = 1
x = np.arange(len(right_dist))/frame_rate

# Plot the first line
axs[0].plot(x, right_dist, marker='o')
axs[0].set_title('Original')

# Plot the second line
axs[1].plot(processed_right_dist)
axs[1].set_title('Processed')

# # Plot the first line
# axs[0].plot(right_dist)
# axs[0].set_title('Right distance')

# # Plot the second line
# axs[1].plot(back_dist)
# axs[1].set_title('Back distance')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
