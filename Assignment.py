import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from pydub import AudioSegment

# Load MP3 file using pydub
audio = AudioSegment.from_mp3("hello.mp3")

# Convert audio to raw data (numpy array)
data = np.array(audio.get_array_of_samples())

# If stereo, take one channel
if audio.channels == 2:
    data = data[::2]

# Sampling rate of the audio file
sample_rate = audio.frame_rate

# Number of samples in the audio file
N = len(data)

# Perform Fourier Transform using FFT
yf = fft(data)

# Frequency bins
xf = fftfreq(N, 1 / sample_rate)

# Plot the waveform of the original audio
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(np.arange(N) / sample_rate, data)
plt.title("Waveform of the Audio")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Plot the Fourier Transform (magnitude spectrum)
plt.subplot(2, 1, 2)
plt.plot(xf[:N // 2], np.abs(yf[:N // 2]) / N)
plt.title("Fourier Transform - Magnitude Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid()

plt.tight_layout()
plt.show()
