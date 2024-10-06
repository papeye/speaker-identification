import shutil
import os
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
