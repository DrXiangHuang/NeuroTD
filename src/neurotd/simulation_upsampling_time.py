import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, resample


# Function for low-pass filtering
def lowpass_filter(data, sampling_rate, cutoff_freq, order=4):
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


# Simulation parameters
f_y = 120  # Original sampling rate of y
f_x = 4000  # Target sampling rate for upsampling
duration = 2  # seconds
t_y = np.arange(0, duration, 1 / f_y)  # Time vector for original y

# Frequencies of the test signal (Hz)
f1, f2, f3 = 5, 50, 300

# Generate a test signal y with multiple frequencies
y = np.sin(2 * np.pi * f1 * t_y) + 0.5 * np.sin(2 * np.pi * f2 * t_y) + 0.2 * np.sin(2 * np.pi * f3 * t_y)

# Adjust cutoff frequency slightly below the Nyquist limit
cutoff_frequency = 0.9 * (f_y / 2)

# Apply low-pass filtering to y
filtered_y = lowpass_filter(y, f_y, cutoff_frequency)


# Function to upsample using time-domain resampling
def upsample_time_domain(signal, original_rate, target_rate):
    # Calculate the number of target samples
    num_samples = int(len(signal) * (target_rate / original_rate))
    # Use scipy's resample to interpolate the signal to the new sampling rate
    upsampled_signal = resample(signal, num_samples)
    return upsampled_signal


# Upsample y and filtered_y using the time-domain approach
upsampled_y_time = upsample_time_domain(y, f_y, f_x)
upsampled_filtered_y_time = upsample_time_domain(filtered_y, f_y, f_x)

# Time vector for the upsampled signals in the time-domain approach
t_x_time = np.linspace(0, duration, len(upsampled_y_time))

# Plot the original, filtered, and upsampled signals for comparison
plt.figure(figsize=(14, 8))

# Original y
plt.subplot(3, 1, 1)
plt.plot(t_y, y, label="Original y")
plt.title("Original y (120 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Upsampled without filtering using time-domain method
plt.subplot(3, 1, 2)
plt.plot(t_x_time, upsampled_y_time, label="Upsampled y without filtering (Time Domain)", alpha=0.7)
plt.title("Upsampled y (4000 Hz) without Low-Pass Filtering (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Upsampled with filtering using time-domain method
plt.subplot(3, 1, 3)
plt.plot(t_x_time, upsampled_filtered_y_time, label="Upsampled y with filtering (Time Domain)", alpha=0.7)
plt.title("Upsampled y (4000 Hz) with Low-Pass Filtering (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()
