import time
import os

from config import Config
from helpers import move_base_data_to_proper_folders, printPrettyDict
from data_preprocessing.audio_cutter import AudioCutter, cut_all_into_segments
from data_preprocessing.noise_preparator import prepareNoise
from data_preprocessing.dataset_generator import (
    generate_train_valid_ds,
    generate_test_ds,
)
from nnmodel import NNModel
from training_type import TrainingType
import numpy as np

""" Flags for execution control"""
# TRAINING_TYPE = TrainingType.PREPARE_DATA_AND_TRAIN
# TRAINING_TYPE = TrainingType.TRAIN_ONLY
TRAINING_TYPE = TrainingType.NO_TRAINING

ADD_NOISE_TO_TRAINING_DATA = False
PREPARE_TEST_DATA = True


def main():
    train_data_dir = "example_data/train_data"
    test_data_dir = "example_data/test_data"

    if TRAINING_TYPE.prepareTrainData:
        move_base_data_to_proper_folders()  # TODO Remove this method - it's obsolete if we use already divided data

        cut_all_into_segments(train_data_dir, Config.dataset_train_audio)

        noises = prepareNoise() if ADD_NOISE_TO_TRAINING_DATA else None

    nn_model = NNModel()

    if TRAINING_TYPE.train:
        train_ds, valid_ds = generate_train_valid_ds(noises)
        nn_model.train(train_ds, valid_ds)
    else:
        nn_model.load()

    if PREPARE_TEST_DATA:
        for file in os.listdir(test_data_dir):
            path = os.path.join(test_data_dir, file)
            AudioCutter(path, Config.dataset_test).cut()

    correctly_identyfied = 0

    for dir in os.listdir(Config.dataset_test):
        path = os.path.join(Config.dataset_test, dir)

        test_ds = generate_test_ds(path, dir)

        predicted_speaker, certainty_measure, speaker_labels = nn_model.predict(test_ds)
        if predicted_speaker == dir:
            correctly_identyfied += 1

        print(f"\n Correct speaker: {dir}, predicted speaker is {predicted_speaker}")

        max_prediction = np.max(certainty_measure)

        for i in range(len(certainty_measure)):
            if certainty_measure[i] > 5:
                if certainty_measure[i] == max_prediction and speaker_labels[i] == dir:
                    print(
                        f"\033[1;32;40m {speaker_labels[i]}: {certainty_measure[i]:.2f}% \033[0m"
                    )
                elif (
                    certainty_measure[i] == max_prediction and speaker_labels[i] != dir
                ):
                    print(
                        f"\033[1;31;40m {speaker_labels[i]}: {certainty_measure[i]:.2f} %\033[0m"
                    )
                else:
                    print(f"{speaker_labels[i]}: {certainty_measure[i]:.2f}%")

    print(
        f"\n Correctly identified speakers: {correctly_identyfied} out of {len(os.listdir(Config.dataset_test))}"
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time} seconds")
