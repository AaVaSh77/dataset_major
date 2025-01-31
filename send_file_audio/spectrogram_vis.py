# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import wavfile
# from scipy.signal import spectrogram
# import librosa.display

# # Load the two audio files
# audio_path_1 = '/home/aavash/Documents/aavashing/AaVash/audiosteg/dataset2/timit/val2/1.wav'  # Replace with your .wav file path
# audio_path_2 = '/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego.wav'  # Replace with your .wav file path

# # Load audio using scipy for both files
# sample_rate_1, audio_data_1 = wavfile.read(audio_path_1)
# sample_rate_2, audio_data_2 = wavfile.read(audio_path_2)

# # If stereo, convert to mono by averaging channels for both files
# if len(audio_data_1.shape) == 2:
#     audio_data_1 = audio_data_1.mean(axis=1)
# if len(audio_data_2.shape) == 2:
#     audio_data_2 = audio_data_2.mean(axis=1)

# # Time axis for waveform plot
# time_1 = np.arange(0, len(audio_data_1)) / sample_rate_1
# time_2 = np.arange(0, len(audio_data_2)) / sample_rate_2

# # Set up the figure to display side by side with adjusted spacing
# fig, axs = plt.subplots(4, 2, figsize=(18, 30), gridspec_kw={'hspace': 1.2, 'wspace': 0.5})

# # Plot 1: Waveform
# axs[0, 0].plot(time_1, audio_data_1, color='blue')
# axs[0, 0].set_title('Waveform (File 1)', fontsize=14)
# axs[0, 0].set_xlabel('Time [s]', fontsize=12)
# axs[0, 0].set_ylabel('Amplitude', fontsize=12)

# axs[0, 1].plot(time_2, audio_data_2, color='blue')
# axs[0, 1].set_title('Waveform (File 2)', fontsize=14)
# axs[0, 1].set_xlabel('Time [s]', fontsize=12)
# axs[0, 1].set_ylabel('Amplitude', fontsize=12)

# # Generate spectrogram for both files
# frequencies_1, times_1, Sxx_1 = spectrogram(audio_data_1, fs=sample_rate_1, nperseg=1024)
# frequencies_2, times_2, Sxx_2 = spectrogram(audio_data_2, fs=sample_rate_2, nperseg=1024)

# # Convert amplitude to decibels for better visualization
# Sxx_db_1 = 10 * np.log10(Sxx_1 + 1e-10)
# Sxx_db_2 = 10 * np.log10(Sxx_2 + 1e-10)

# # Plot 2: Spectrogram
# spectrogram_plot_1 = axs[1, 0].pcolormesh(times_1, frequencies_1, Sxx_db_1, shading='gouraud', cmap='viridis')
# axs[1, 0].set_title('Spectrogram (File 1)', fontsize=14)
# axs[1, 0].set_xlabel('Time [s]', fontsize=12)
# axs[1, 0].set_ylabel('Frequency [Hz]', fontsize=12)
# fig.colorbar(spectrogram_plot_1, ax=axs[1, 0], label='Intensity (dB)')

# spectrogram_plot_2 = axs[1, 1].pcolormesh(times_2, frequencies_2, Sxx_db_2, shading='gouraud', cmap='viridis')
# axs[1, 1].set_title('Spectrogram (File 2)', fontsize=14)
# axs[1, 1].set_xlabel('Time [s]', fontsize=12)
# axs[1, 1].set_ylabel('Frequency [Hz]', fontsize=12)
# fig.colorbar(spectrogram_plot_2, ax=axs[1, 1], label='Intensity (dB)')

# # Plot 3: Power Spectral Density (PSD)
# axs[2, 0].psd(audio_data_1, NFFT=1024, Fs=sample_rate_1, noverlap=512, color='purple')
# axs[2, 0].set_title('Power Spectral Density (File 1)', fontsize=14)
# axs[2, 0].set_xlabel('Frequency [Hz]', fontsize=12)
# axs[2, 0].set_ylabel('Power/Frequency [dB/Hz]', fontsize=12)

# axs[2, 1].psd(audio_data_2, NFFT=1024, Fs=sample_rate_2, noverlap=512, color='purple')
# axs[2, 1].set_title('Power Spectral Density (File 2)', fontsize=14)
# axs[2, 1].set_xlabel('Frequency [Hz]', fontsize=12)
# axs[2, 1].set_ylabel('Power/Frequency [dB/Hz]', fontsize=12)

# # Load audio using librosa for both files
# y_1, sr_1 = librosa.load(audio_path_1, sr=None)
# y_2, sr_2 = librosa.load(audio_path_2, sr=None)

# # Plot 4: Spectrogram using librosa
# D_1 = librosa.amplitude_to_db(np.abs(librosa.stft(y_1)), ref=np.max)
# D_2 = librosa.amplitude_to_db(np.abs(librosa.stft(y_2)), ref=np.max)

# axs[3, 0].imshow(D_1, aspect='auto', origin='lower', cmap='inferno')
# axs[3, 0].set_title('Spectrogram (Librosa, File 1)', fontsize=14)
# axs[3, 0].set_xlabel('Time [s]', fontsize=12)
# axs[3, 0].set_ylabel('Frequency [Hz]', fontsize=12)
# fig.colorbar(axs[3, 0].images[0], ax=axs[3, 0], format='%+2.0f dB')

# axs[3, 1].imshow(D_2, aspect='auto', origin='lower', cmap='inferno')
# axs[3, 1].set_title('Spectrogram (Librosa, File 2)', fontsize=14)
# axs[3, 1].set_xlabel('Time [s]', fontsize=12)
# axs[3, 1].set_ylabel('Frequency [Hz]', fontsize=12)
# fig.colorbar(axs[3, 1].images[0], ax=axs[3, 1], format='%+2.0f dB')

# # Adjust layout to prevent overlapping
# plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
# plt.tight_layout(pad=2.0)  # Adjust padding between plots
# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import wavfile
# from scipy.signal import spectrogram
# import librosa.display

# # Load audio files
# sample_rate_1, audio_data_1 = wavfile.read('/home/aavash/Documents/aavashing/AaVash/audiosteg/cover2.wav')
# sample_rate_2, audio_data_2 = wavfile.read('/home/aavash/Documents/aavashing/AaVash/audiosteg/secret4.wav')

# # Convert to mono if stereo
# if len(audio_data_1.shape) == 2:
#     audio_data_1 = audio_data_1.mean(axis=1)
# if len(audio_data_2.shape) == 2:
#     audio_data_2 = audio_data_2.mean(axis=1)

# # Match duration
# min_length = min(len(audio_data_1), len(audio_data_2))
# audio_data_1 = audio_data_1[:min_length]
# audio_data_2 = audio_data_2[:min_length]

# # Time axis
# time = np.arange(0, min_length) / sample_rate_1

# # Plot setup
# fig, axs = plt.subplots(2, 2, figsize=(18, 30), gridspec_kw={'hspace': 1.2, 'wspace': 0.5})

# # Waveform Plot
# axs[0, 0].plot(time, audio_data_1, color='blue')
# axs[0, 0].set_title('Waveform of Cover Audio')
# axs[1, 0].plot(time, audio_data_2, color='blue')
# axs[1, 0].set_title('Waveform of Secret Audio')

# # # Spectrogram
# # frequencies_1, times_1, Sxx_1 = spectrogram(audio_data_1, fs=sample_rate_1, nperseg=1024)
# # frequencies_2, times_2, Sxx_2 = spectrogram(audio_data_2, fs=sample_rate_2, nperseg=1024)
# # axs[1, 0].pcolormesh(times_1, frequencies_1, 10 * np.log10(Sxx_1 + 1e-10), shading='gouraud')
# # axs[1, 0].set_title('Spectrogram (File 1)')
# # axs[1, 1].pcolormesh(times_2, frequencies_2, 10 * np.log10(Sxx_2 + 1e-10), shading='gouraud')
# # axs[1, 1].set_title('Spectrogram (File 2)')

# # # PSD
# # axs[2, 0].psd(audio_data_1, NFFT=1024, Fs=sample_rate_1, noverlap=512)
# # axs[2, 0].set_title('PSD (File 1)')
# # axs[2, 1].psd(audio_data_2, NFFT=1024, Fs=sample_rate_2, noverlap=512)
# # axs[2, 1].set_title('PSD (File 2)')

# # # Librosa Spectrogram
# # y_1, sr_1 = librosa.load('/home/aavash/Documents/aavashing/AaVash/audiosteg/cover2.wav', sr=None, duration=min_length / sample_rate_1)
# # y_2, sr_2 = librosa.load('/home/aavash/Documents/aavashing/AaVash/audiosteg/secret4.wav', sr=None, duration=min_length / sample_rate_2)
# # D_1 = librosa.amplitude_to_db(np.abs(librosa.stft(y_1)), ref=np.max)
# # D_2 = librosa.amplitude_to_db(np.abs(librosa.stft(y_2)), ref=np.max)
# # axs[3, 0].imshow(D_1, aspect='auto', origin='lower', cmap='inferno')
# # axs[3, 0].set_title('Librosa Spectrogram (File 1)')
# # axs[3, 1].imshow(D_2, aspect='auto', origin='lower', cmap='inferno')
# # axs[3, 1].set_title('Librosa Spectrogram (File 2)')

# # Adjust layout
# plt.tight_layout(pad=2.0)
# plt.show()


















# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np
# from IPython.display import Audio

# def visualize_audio(file_path):
#     # Load the audio file
#     y, sr = librosa.load(file_path, sr=None)
    
#     # Plot the waveform
#     plt.figure(figsize=(14, 6))
    
#     plt.subplot(2, 1, 1)
#     librosa.display.waveshow(y, sr=sr)
#     plt.title('Waveform of Enhanced Secret Audio')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
    
#     # Plot the spectrogram
#     plt.subplot(2, 1, 2)
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
#     librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram of Enhanced Secret Audio')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Frequency (Hz)')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Play the audio
#     return Audio(file_path)

# # Example usage:
# audio = visualize_audio('/home/aavash/Documents/aavashing/AaVash/audiosteg/enhanced_audio.wav')
# audio







import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Load recovered audio
audio, sr = librosa.load("final_recovered_original.wav", sr=None)

# Step 1: Adaptive Spectral Subtraction for noise reduction
def spectral_subtract(y, sr, noise_reduce_factor=0.02):
    D = librosa.stft(y)
    magnitude, phase = librosa.magphase(D)
    noise_profile = np.mean(magnitude[:, :10], axis=1, keepdims=True)
    magnitude_subtracted = np.maximum(magnitude - noise_reduce_factor * noise_profile, 0)
    return librosa.istft(magnitude_subtracted * phase)

denoised_audio = spectral_subtract(audio, sr)

# Step 2: Parametric Equalizer (boost speech frequencies)
def apply_eq(y, sr):
    # Boost around 3000 Hz for speech clarity
    def band_boost(y, lowcut, highcut, boost_db):
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(2, [low, high], btype='band')
        boosted = lfilter(b, a, y) * (10**(boost_db / 20))
        return y + boosted

    y = band_boost(y, 300, 3400, 5)  # Boost speech range by +5 dB
    return y

eq_audio = apply_eq(denoised_audio, sr)

# Step 3: Normalize to increase loudness
normalized_audio = eq_audio / np.max(np.abs(eq_audio)) * 0.9

# Save the enhanced audio
sf.write("final_enhanced_audio.wav", normalized_audio, sr)

# Step 4: Plot Spectrogram for comparison
plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(normalized_audio)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of Final Enhanced Audio')
plt.tight_layout()
plt.show()

