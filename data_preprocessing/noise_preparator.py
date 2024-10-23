import os
from config import Config
from pathlib import Path
import tensorflow as tf
from typing import Optional, List


def __prepare_noise() -> str:
    """
    We load all noise samples (which should have been resampled to 16000)
    We split those noise samples to chunks of 16000 samples which correspond to 1 second duration each
    """
    noise_folder = Config.dataset_train_noise

    # Get the list of all noise files
    noise_paths = []
    for subdir in os.listdir(noise_folder):
        subdir_path = Path(noise_folder) / subdir
        if os.path.isdir(subdir_path):
            noise_paths += [
                os.path.join(subdir_path, filepath)
                for filepath in os.listdir(subdir_path)
                if filepath.endswith(".wav")
            ]
    if not noise_paths:
        raise RuntimeError(f"Could not find any files at {noise_folder}")
    print(
        f"Found {len(noise_paths)} files belonging to {len(os.listdir(Config.dataset_train_noise))} directories"
    )

    return noise_paths

#Actually list of tensors??
def __load_noise_sample(path: str) -> Optional[List[tf.Tensor]]:
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == Config.sampling_rate:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / Config.sampling_rate)
        sample = tf.split(sample[: slices * Config.sampling_rate], slices)
        return sample
    else:
        print(f"Sampling rate for {path} is incorrect. Ignoring it")
        return None


def __load_noise(noise_paths: List[str]) -> tf.Tensor:
    noises = []
    for path in noise_paths:
        sample = __load_noise_sample(path)
        if sample:
            noises.extend(sample)
    return tf.stack(noises) #Stack list of tensors into one tensor


def prepareNoise() -> tf.Tensor:
    noise_paths = __prepare_noise()

    noises = __load_noise(noise_paths)

    print("Noises moved to proper folders")

    return noises
