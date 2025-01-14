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


def display_predictions(predictions, correctly_identified):

    for detail in predictions:
        correct_speaker = detail["correct_speaker"]
        predicted_speaker = detail["predicted_speaker"]
        certainty_measure = detail["certainty_measure"]
        max_prediction = np.max(certainty_measure)

        print(
            f"\nCorrect speaker: {correct_speaker}, predicted speaker is {predicted_speaker}"
        )

        for i in range(len(certainty_measure)):
            if certainty_measure[i] > 5:
                if (
                    certainty_measure[i] == max_prediction
                    and predicted_speaker == correct_speaker
                ):
                    print(
                        f"\033[1;32;40m {predicted_speaker}: {certainty_measure[i]:.2f}% \033[0m"
                    )
                elif (
                    certainty_measure[i] == max_prediction
                    and predicted_speaker != correct_speaker
                ):
                    print(
                        f"\033[1;31;40m {predicted_speaker}: {certainty_measure[i]:.2f}% \033[0m"
                    )
                else:
                    print(f"{predicted_speaker}: {certainty_measure[i]:.2f}%")

    print(f"\nCorrectly identified speakers: {correctly_identified}")
