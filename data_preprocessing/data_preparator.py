import os
import shutil
import librosa
from config import Config
from pathlib import Path
import soundfile as sf
import tensorflow as tf


class DataPreparator:
    """
    DataPreparator handles
    1. Sorting files to respective folders (Config.dataset_audio_path, Config.dataset_noise_path)
    2. resampling audio subsegments to {Config.sample_rate}
    """

    def ___move_files_to_proper_folders(self):
        for folder in os.listdir(Config.dataset_root):
            if os.path.isdir(os.path.join(Config.dataset_root, folder)):
                if folder in ["other", "_background_noise_"]:
                    # If folder is one of the folders that contains noise samples,
                    # move it to the `noise` folder
                    shutil.copytree(
                        os.path.join(Config.dataset_root, folder),
                        os.path.join(Config.dataset_train_noise, folder),
                        dirs_exist_ok=True,
                    )
                else:
                    # Otherwise, it should be a speaker folder, then move it to
                    # `audio` folder
                    shutil.copytree(
                        os.path.join(Config.dataset_root, folder),
                        os.path.join(Config.dataset_train_audio, folder),
                        dirs_exist_ok=True,
                    )

    def __resample(self, folder_path):
        files = os.listdir(folder_path)
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if os.path.exists(file_path):
                print(file_path)

            y, sr = librosa.load(file_path)
            resampled_y = librosa.resample(
                y, orig_sr=sr, target_sr=Config.sampling_rate
            )

            sf.write(file_path, resampled_y, samplerate=Config.sampling_rate)

        print(f"Resampled {len(files)} file in {folder_path} to {Config.sampling_rate}!")

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
            "Found {} files belonging to {} directories".format(
                len(noise_paths), len(os.listdir(Config.dataset_train_noise))
            )
        )

        for folder in os.listdir(noise_folder):
            self.__resample(os.path.join(noise_folder, folder))
            print(folder)

        return noise_paths

    def __prepare_new_speaker(self, audio_name):
        new_subsegments_folder = os.path.join(Config.dataset_train_audio, audio_name)
        self.__resample(new_subsegments_folder)

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
            print("Sampling rate for {} is incorrect. Ignoring it".format(path))
            return None

    def __load_noise(self, noise_paths):
        noises = []
        for path in noise_paths:
            sample = self.__load_noise_sample(path)
            if sample:
                noises.extend(sample)
        return tf.stack(noises)

    def prepare(self, audio_name):
        self.___move_files_to_proper_folders()
        noise_paths = self.__prepare_noise()
        print("aaaaaaaaaaaaaaaaaaaaaaa")
        self.__prepare_new_speaker(audio_name)
        print("bbbbbbbbbbbbbbbbbbbbbbbbb")
        return self.__load_noise(noise_paths)
