import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft, fftfreq

# Function for low-pass filtering
def lowpass_filter(data, sampling_rate, cutoff_freq, order=4):
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Simulation parameters
f_y = 120  # Original sampling rate of y
f_x = 4000  # Target sampling rate for upsampling
duration = 2  # seconds
t_y = np.arange(0, duration, 1/f_y)  # Time vector for original y

# Generate a test signal y with multiple frequencies:
f1, f2, f3 = 5, 50, 60  # Frequencies of the test signal (Hz)
y = np.sin(2 * np.pi * f1 * t_y) + 0.5 * np.sin(2 * np.pi * f2 * t_y) + 0.2 * np.sin(2 * np.pi * f3 * t_y)

# Adjust cutoff frequency slightly below the Nyquist limit
cutoff_frequency = 0.9 * (f_y / 2)

# Apply low-pass filtering to y
filtered_y = lowpass_filter(y, f_y, cutoff_frequency)

# Function to upsample using DFT and zero-padding
def upsample_via_dft(signal, target_length):
    signal_dft = fft(signal, target_length)  # DFT with zero-padding
    # Scale the magnitudes to match the new sampling rate
    scaled_signal_dft = signal_dft * (f_x / f_y)
    upsampled_signal = ifft(scaled_signal_dft).real  # Inverse DFT to get the time-domain signal
    return upsampled_signal

# Upsample y and filtered_y
N_y = len(y)
N_x = int(N_y * (f_x / f_y))  # Target length for upsampling
upsampled_y = upsample_via_dft(y, N_x)
upsampled_filtered_y = upsample_via_dft(filtered_y, N_x)

# Time vector for the upsampled signals
t_x = np.arange(0, duration, 1/f_x)

# Plot the original, filtered, and upsampled signals for comparison
plt.figure(figsize=(14, 8))

# Original y
plt.subplot(3, 1, 1)
plt.plot(t_y, y, label='Original y')
plt.title('Original y (120 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Upsampled without filtering
plt.subplot(3, 1, 2)
plt.plot(t_x, upsampled_y, label='Upsampled y without filtering', alpha=0.7)
plt.title('Upsampled y (4000 Hz) without Low-Pass Filtering')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Upsampled with filtering
plt.subplot(3, 1, 3)
plt.plot(t_x, upsampled_filtered_y, label='Upsampled y with filtering', alpha=0.7)
plt.title('Upsampled y (4000 Hz) with Low-Pass Filtering')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Function to upsample using time-domain resampling
def upsample_time_domain(signal, original_rate, target_rate):
    # Calculate the number of target samples
    num_samples = int(len(signal) * (target_rate / original_rate))
    # Use scipy's resample to interpolate the signal to the new sampling rate
    upsampled_signal = resample(signal, num_samples)
    return upsampled_signal

plt.tight_layout()
plt.show()
