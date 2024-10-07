import shutil
import os
import librosa
import soundfile as sf

from config import Config


class Helpers:
    @staticmethod
    def move_files(source, target):
        """Copies all files from source to target directory"""

        if not os.path.exists(target):
            os.makedirs(target)

        for filename in os.listdir(source):
            source_path = os.path.join(source, filename)
            destination_path = os.path.join(target, filename)
            shutil.copy(source_path, destination_path)

    # TODO Remove this method - it's obsolety if we use already divided data
    @staticmethod
    def move_base_data_to_proper_folders():
        """Divides base data into audio and noise folders"""

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

    @staticmethod
    def resampleAll(folder_path):
        """Resamples all files in the folder_path and all subfolders to Config.sampling_rate"""
        files = os.listdir(folder_path)

        for file in files:
            file_path = os.path.join(folder_path, file)

            if os.path.isdir(file_path):
                Helpers.resampleAll(file_path)
                continue

            y, sr = librosa.load(file_path)
            resampled_y = librosa.resample(
                y, orig_sr=sr, target_sr=Config.sampling_rate
            )

            sf.write(file_path, resampled_y, samplerate=Config.sampling_rate)

        print(
            f"Resampled {len(files)} files in {folder_path} to {Config.sampling_rate}!"
        )
