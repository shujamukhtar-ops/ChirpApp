import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

# ---------------------------
# Band-pass filter function
# ---------------------------
def bandpass_filter(data, fs, lowcut=18000, highcut=20000, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# ---------------------------
# File paths
# ---------------------------
input_dir = "/home/think-mint/ChirApp-Recordings/"
file_to_convert = "eating.wav"
output_dir = "/home/think-mint/Converted-WAV-Recordings"
os.makedirs(output_dir, exist_ok=True)

input_path = os.path.join(input_dir, file_to_convert)

# ---------------------------
# Read and process the file
# ---------------------------
fs, data = wav.read(input_path)
data = data.astype(float)

# Convert to mono if stereo
if data.ndim > 1:
    data = np.mean(data, axis=1)

# Create time vector
t = np.arange(len(data)) / fs

# Apply band-pass filter (18–20 kHz)
filtered = bandpass_filter(data, fs)

# FFT for frequency analysis
freq_domain = np.fft.fft(data)
freqs = np.fft.fftfreq(len(data), 1 / fs)


plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, data, color='gray')
plt.title("Original Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(t, filtered, color='blue')
plt.title("Filtered Signal (18–20 kHz)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.specgram(filtered, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
plt.title("Spectrogram (Filtered Signal)")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label="Intensity [dB]")

plt.tight_layout()
plt.show()

# ---------------------------
# Save filtered data as .pmc
# ---------------------------
pmc_path = os.path.join(output_dir, file_to_convert.replace(".wav", ".pmc"))
filtered.astype(np.float32).tofile(pmc_path)

print(f"Processed: {file_to_convert} → {pmc_path}")
