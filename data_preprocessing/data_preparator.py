import os

from helpers import Helpers
from config import Config
from pathlib import Path

import tensorflow as tf


class NoisePreparator:
    """
    DataPreparator handles
    1. Sorting files to respective folders (Config.dataset_audio_path, Config.dataset_noise_path)
    2. resampling audio subsegments to {Config.sample_rate}
    """

    def __prepare_noise(self):
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

        for folder in os.listdir(noise_folder):
            Helpers.resampleAll(os.path.join(noise_folder, folder))

        return noise_paths

    # Split noise into chunks of 16,000 steps each
    def __load_noise_sample(self, path):
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

    def __load_noise(self, noise_paths):
        noises = []
        for path in noise_paths:
            sample = self.__load_noise_sample(path)
            if sample:
                noises.extend(sample)
        return tf.stack(noises)

    def prepare(self):
        noise_paths = self.__prepare_noise()

        return self.__load_noise(noise_paths)
