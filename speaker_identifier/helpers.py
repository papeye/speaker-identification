import shutil
import os
import numpy as np

from .config import Config


def move_files(source: str, target: str) -> None:
    """Copies all files from source to target directory"""

    if not os.path.exists(target):
        os.makedirs(target)

    for filename in os.listdir(source):
        source_path = os.path.join(source, filename)
        destination_path = os.path.join(target, filename)
        shutil.copy(source_path, destination_path)


def move_base_data_to_proper_folders(no_base_speakers_to_move: int) -> None:
    """Divides base data into audio and noise folders"""

    processed_count = 0

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
                # Process speaker folders
                if processed_count < no_base_speakers_to_move:
                    shutil.copytree(
                        os.path.join(Config.dataset_root, folder),
                        os.path.join(Config.dataset_train_audio, folder),
                        dirs_exist_ok=True,
                    )
                    processed_count += 1
                else:
                    # Stop processing if we've reached the desired number of folders
                    break


def remove_dir(dir: str):
    if os.path.exists(dir):
        shutil.rmtree(dir)
