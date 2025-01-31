import os
import shutil
from pydub import AudioSegment
from pydub.utils import make_chunks

# Paths to original model and validation folder
VALIDATION_FOLDER = "dataset2/timit/val2"
MODEL_SCRIPT = "main.py"
OUTPUT_FOLDER = "final_output/output_combined"
RUN_DIR = "/home/aavash/Documents/experiment_output"
CHECKPOINT = "/home/aavash/Documents/experiment_folder/experiment_output-3/ckpt/154_epoch"
TEMP_OUTPUT_FOLDER = "temp/output_chunks"

# Function to split audio into 3-second chunks
def split_audio(audio_path, chunk_length_ms=3000):
    """
    Splits the audio into chunks of the specified duration.

    Args:
        audio_path (str): Path to the audio file to split.
        chunk_length_ms (int): Length of each chunk in milliseconds (default: 3000ms).

    Returns:
        list[AudioSegment]: List of audio chunks.
    """
    audio = AudioSegment.from_file(audio_path)
    if len(audio) < chunk_length_ms:
        raise ValueError("Audio length is shorter than the specified chunk size.")
    chunks = make_chunks(audio, chunk_length_ms)
    print(f"Split audio into {len(chunks)} chunks of {chunk_length_ms / 1000} seconds each.")
    return chunks

    print(f"Split audio into {len(trimmed_chunks)} trimmed chunks of {chunk_length_ms / 1000} seconds each.")
    return trimmed_chunks


def get_latest_experiment_output(base_dir):
    """Find the latest experiment output folder created by the model."""
    experiment_dirs = [d for d in os.listdir(base_dir) if d.startswith("experiment_output-")]
    if not experiment_dirs:
        raise Exception("No experiment output folders found!")
    latest_dir = sorted(experiment_dirs, key=lambda x: int(x.split('-')[-1]))[-1]
    return os.path.join(base_dir, latest_dir)

def delete_experiment_output(base_dir):
    """Delete all experiment output folders created by the model."""
    experiment_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("experiment_output-")]
    for dir_path in experiment_dirs:
        shutil.rmtree(dir_path, ignore_errors=True)

# Function to save chunks to specified folder
def save_chunks(chunks, folder_path, base_name):
    """
    Saves the audio chunks to the specified folder.

    Args:
        chunks (list[AudioSegment]): List of audio chunks to save.
        folder_path (str): Path to the folder where chunks will be saved.
        base_name (str): Base name for each chunk file.

    Returns:
        list[str]: List of paths to the saved chunks.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(folder_path, f"{base_name}_{i}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

    return chunk_paths

# Function to process each chunk pair with the model
def process_with_model(cover_chunk_paths, secret_chunk_paths):
    """
    Processes audio chunk pairs with the model and collects outputs.

    Args:
        cover_chunk_paths (list[str]): Paths to cover audio chunks.
        secret_chunk_paths (list[str]): Paths to secret audio chunks.

    Returns:
        tuple: Lists of stego chunks, recovered original chunks, and recovered GL chunks.
    """
    stego_chunks = []
    recovered_original_chunks = []
    recovered_gl_chunks = []

    if not os.path.exists(TEMP_OUTPUT_FOLDER):
        os.makedirs(TEMP_OUTPUT_FOLDER)

    for i, (cover_chunk, secret_chunk) in enumerate(zip(cover_chunk_paths, secret_chunk_paths)):
        # Move chunks to validation folder
        shutil.copy(cover_chunk, os.path.join(VALIDATION_FOLDER, "cover.wav"))
        shutil.copy(secret_chunk, os.path.join(VALIDATION_FOLDER, "secret.wav"))

        # Run the original model
        os.system(f"python {MODEL_SCRIPT} --mode sample --train_path dataset2/timit/train --val_path dataset2/timit/val2 --test_path dataset2/timit/test --dataset timit --run_dir {RUN_DIR} --block_type relu --lr 0.0007 --batch_size 16 --n_pairs 5000 --model_type n_msg --enc_n_layers 3 --dec_c_n_layers 4 --save_model_every 5 --sample_every 5 --num_workers 10 --num_iters 500 --load_ckpt {CHECKPOINT}")

        # Get the latest experiment output folder
        latest_output_folder = get_latest_experiment_output("/home/aavash/Documents")

        # Collect the output files from the correct subpath
        stego_chunk_path = os.path.join(TEMP_OUTPUT_FOLDER, f"carrier_embedded_{i}.wav")
        recovered_original_chunk_path = os.path.join(TEMP_OUTPUT_FOLDER, f"recovered_msg_original_phase_{i}.wav")
        recovered_gl_chunk_path = os.path.join(TEMP_OUTPUT_FOLDER, f"recovered_msg_gl_phase_{i}.wav")

        shutil.copy(os.path.join(latest_output_folder, "samples/0/0_cover_carrier_embedded.wav"), stego_chunk_path)
        shutil.copy(os.path.join(latest_output_folder, "samples/0/0_secret_msg_recovered_orig_phase.wav"), recovered_original_chunk_path)
        shutil.copy(os.path.join(latest_output_folder, "samples/0/0_secret_msg_recovered_gl_phase.wav"), recovered_gl_chunk_path)

        stego_chunks.append(AudioSegment.from_file(stego_chunk_path))
        recovered_original_chunks.append(AudioSegment.from_file(recovered_original_chunk_path))
        recovered_gl_chunks.append(AudioSegment.from_file(recovered_gl_chunk_path))

        # Delete the experiment output folder for this chunk
        delete_experiment_output("/home/aavash/Documents")

    return stego_chunks, recovered_original_chunks, recovered_gl_chunks

# Function to combine chunks into complete audio files
def combine_chunks(chunks, output_path):
    """
    Combines audio chunks into a single file and saves it.

    Args:
        chunks (list[AudioSegment]): List of audio chunks to combine.
        output_path (str): Path to save the combined audio file.
    """
    combined = sum(chunks)
    combined.export(output_path, format="wav")

# Main function
def main(cover_audio_path, secret_audio_path):
    """
    Main function to handle the audio steganography process.

    Args:
        cover_audio_path (str): Path to the cover audio file.
        secret_audio_path (str): Path to the secret audio file.
    """
    # Step 1: Split cover and secret audio
    cover_chunks = split_audio(cover_audio_path)
    secret_chunks = split_audio(secret_audio_path)

    # Step 2: Save chunks to temporary folder
    cover_chunk_paths = save_chunks(cover_chunks, "temp/cover_chunks", "cover")
    secret_chunk_paths = save_chunks(secret_chunks, "temp/secret_chunks", "secret")

    # Step 3: Process chunks with the model
    stego_chunks, recovered_original_chunks, recovered_gl_chunks = process_with_model(cover_chunk_paths, secret_chunk_paths)

    # Step 4: Combine output chunks into final audio files
    combine_chunks(stego_chunks, "final_stego.wav")
    combine_chunks(recovered_original_chunks, "final_recovered_original.wav")
    combine_chunks(recovered_gl_chunks, "final_recovered_gl.wav")

if __name__ == "__main__":
    cover_audio_path = "dataset2/timit/val2/1.wav"
    secret_audio_path = "dataset2/timit/val2/2.wav"
    main(cover_audio_path, secret_audio_path)

