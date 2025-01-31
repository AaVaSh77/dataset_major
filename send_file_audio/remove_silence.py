# from pydub import AudioSegment
# from pydub.silence import split_on_silence

# def remove_silence(input_audio_path, output_audio_path, silence_thresh=-40, min_silence_len=1000):
#     """
#     Removes silent parts from the audio file.
    
#     Parameters:
#     - input_audio_path: Path to the input audio file.
#     - output_audio_path: Path to save the processed audio file without silence.
#     - silence_thresh: Silence threshold in dB (default -40 dB).
#     - min_silence_len: Minimum length of silence in milliseconds to be considered (default 1000 ms).
#     """
#     # Load the audio file
#     audio = AudioSegment.from_file(input_audio_path)
    
#     # Split the audio into non-silent parts based on silence threshold
#     chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
#     # Concatenate the non-silent chunks back together
#     processed_audio = AudioSegment.empty()
#     for chunk in chunks:
#         processed_audio += chunk
    
#     # Export the processed audio to a new file
#     processed_audio.export(output_audio_path, format="wav")
#     print(f"Processed audio saved to {output_audio_path}")

# # Example usage
# input_audio_path = '/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego.wav'  # Replace with the path to your audio file
# output_audio_path = '/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego_no_noise.wav'    # Replace with the desired output path

# remove_silence(input_audio_path, output_audio_path)



# from pydub import AudioSegment
# from pydub.silence import split_on_silence

# # Load the audio file
# audio_path = '/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego.wav'   # Replace with your audio file path
# audio = AudioSegment.from_wav(audio_path)

# # Split the audio into chunks based on silence
# chunks = split_on_silence(audio, 
#                           min_silence_len=1000,  # Minimum length of silence to split (in ms)
#                           silence_thresh=-40)    # Silence threshold in dB

# # Join all the chunks together to create a smooth audio file
# smooth_audio = AudioSegment.empty()
# for chunk in chunks:
#     smooth_audio += chunk

# # Save the smooth audio to a new file
# smooth_audio.export('/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego_no_noise.wav' , format='wav')




# import numpy as np
# from scipy.io import wavfile
# import os

# def apply_crossfade(audio_data, sample_rate, fade_duration_ms=50):
#     """
#     Applies a crossfade to smooth transitions between segments.

#     Parameters:
#         audio_data (np.ndarray): Audio data array.
#         sample_rate (int): Sample rate of the audio.
#         fade_duration_ms (int): Duration of the crossfade in milliseconds.

#     Returns:
#         np.ndarray: Smoothed audio data.
#     """
#     fade_samples = int((fade_duration_ms / 1000) * sample_rate)
#     num_segments = len(audio_data) // sample_rate

#     for i in range(1, num_segments):
#         # Define the fade-out and fade-in regions
#         start_idx = i * sample_rate - fade_samples
#         end_idx = i * sample_rate

#         fade_out = np.linspace(1, 0, fade_samples)
#         fade_in = np.linspace(0, 1, fade_samples)

#         # Apply crossfade
#         audio_data[start_idx:end_idx] = (
#             audio_data[start_idx:end_idx] * fade_out
#             + audio_data[end_idx:end_idx + fade_samples] * fade_in
#         )

#     return audio_data

# # Load the audio file
# file_path = "/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego.wav"  # Replace with your audio file path
# output_path = "/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego_no_noise.wav"

# sample_rate, audio_data = wavfile.read(file_path)

# # Ensure the audio is mono for simplicity
# if audio_data.ndim > 1:
#     audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)

# # Apply crossfade smoothing
# smoothed_audio = apply_crossfade(audio_data, sample_rate)

# # Save the smoothed audio
# wavfile.write(output_path, sample_rate, smoothed_audio)
# print(f"Smoothed audio saved to {output_path}")

# # Verify file creation
# if os.path.exists(output_path):
#     print("File created successfully!")
# else:
#     print("Error in creating the file.")


# import numpy as np
# from scipy.io import wavfile
# import os

# def apply_crossfade(audio_data, sample_rate, fade_duration_ms=50):
#     """
#     Applies a crossfade to smooth transitions between segments.

#     Parameters:
#         audio_data (np.ndarray): Audio data array.
#         sample_rate (int): Sample rate of the audio.
#         fade_duration_ms (int): Duration of the crossfade in milliseconds.

#     Returns:
#         np.ndarray: Smoothed audio data.
#     """
#     fade_samples = int((fade_duration_ms / 1000) * sample_rate)
#     segment_duration = sample_rate
#     num_segments = len(audio_data) // segment_duration

#     for i in range(1, num_segments):
#         # Define the fade-out and fade-in regions
#         start_idx = i * segment_duration - fade_samples
#         end_idx = i * segment_duration

#         if end_idx + fade_samples > len(audio_data):
#             break

#         fade_out = np.linspace(1, 0, fade_samples)
#         fade_in = np.linspace(0, 1, fade_samples)

#         # Apply crossfade
#         audio_data[start_idx:end_idx] = (
#             audio_data[start_idx:end_idx] * fade_out
#             + audio_data[end_idx:end_idx + fade_samples] * fade_in
#         )

#     return audio_data

# def smooth_audio(input_path, output_path, fade_duration_ms=50):
#     """
#     Reads an audio file, applies crossfade smoothing, and saves the result.

#     Parameters:
#         input_path (str): Path to the input audio file.
#         output_path (str): Path to save the smoothed audio file.
#         fade_duration_ms (int): Duration of the crossfade in milliseconds.

#     Returns:
#         None
#     """
#     # Load the audio file
#     sample_rate, audio_data = wavfile.read(input_path)

#     # Ensure the audio is mono for simplicity
#     if audio_data.ndim > 1:
#         audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)

#     # Apply crossfade smoothing
#     smoothed_audio = apply_crossfade(audio_data, sample_rate, fade_duration_ms)

#     # Save the smoothed audio
#     wavfile.write(output_path, sample_rate, smoothed_audio)
#     print(f"Smoothed audio saved to {output_path}")

# # Paths to input and output audio files
# input_path = "/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego.wav"  # Replace with your audio file path
# output_path = "/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego_no_noise_2.wav"

# # Smooth the audio and save
# smooth_audio(input_path, output_path, fade_duration_ms=50)

# # Verify file creation
# if os.path.exists(output_path):
#     print("File created successfully!")
# else:
#     print("Error in creating the file.")







# from pydub import AudioSegment
# from pydub.effects import normalize

# # Load the audio file
# audio_path = '/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego.wav'
# audio = AudioSegment.from_file(audio_path)

# # Break the audio into 1-second segments and normalize each segment
# segment_duration_ms = 1000  # 1 second in milliseconds
# segments = []

# for i in range(0, len(audio), segment_duration_ms):
#     segment = audio[i:i + segment_duration_ms]
#     segment = normalize(segment)  # Normalize to smooth out volume differences
#     segments.append(segment)

# # Combine the segments back into a single audio file
# fixed_audio = sum(segments)

# # Export the processed audio
# output_path = '/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego_no_noise_2.wav'
# fixed_audio.export(output_path, format="wav")

# print(f"Processed audio saved to: {output_path}")


import numpy as np
from scipy.io import wavfile

def remove_silence(input_path, output_path, silence_threshold=1e-4):
    # Load the audio file
    sampling_rate, audio_data = wavfile.read(input_path)
    
    # Normalize audio data if in integer format
    if audio_data.dtype == np.int16:
        audio_data = audio_data / 32768.0
    
    # Detect non-silent segments
    non_silent_indices = np.where(np.abs(audio_data) > silence_threshold)[0]
    if len(non_silent_indices) == 0:
        print("No non-silent audio detected.")
        return
    
    # Trim the silent parts
    start_index = non_silent_indices[0]
    end_index = non_silent_indices[-1]
    trimmed_audio = audio_data[start_index:end_index + 1]
    
    # Save the processed audio
    if audio_data.dtype == np.int16:
        trimmed_audio = (trimmed_audio * 32768).astype(np.int16)
    
    wavfile.write(output_path, sampling_rate, trimmed_audio)
    print(f"Processed audio saved to: {output_path}")

# Paths
input_path = "/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego.wav"  # Replace with your audio file path
output_path = "/home/aavash/Documents/aavashing/AaVash/audiosteg/final_stego_no_noise_2.wav"

# Call the function to process the audio
remove_silence(input_path, output_path)

