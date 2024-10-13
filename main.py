import time

from data_preprocessing.audio_cutter import cut_all_into_segments
from config import Config
from data_preprocessing.noise_preparator import prepare_noise
from data_preprocessing.dataset_generator import (
    generate_train_valid_ds,
    generate_test_ds,
)
from nnmodel import NNModel
from helpers import move_base_data_to_proper_folders, print_pretty_dict
from training_type import TrainingType

""" Flags for execution control"""
# TRAINING_TYPE = TrainingType.PREPARE_DATA_AND_TRAIN
# TRAINING_TYPE = TrainingType.TRAIN_ONLY
TRAINING_TYPE = TrainingType.NO_TRAINING

ADD_NOISE_TO_TRAINING_DATA = False
PREPARE_TEST_DATA = False


def main():
    train_data_dir = "example_data/train_data"
    test_data_dir = "example_data/test_data"

    if TRAINING_TYPE.prepare_train_data():
        move_base_data_to_proper_folders()  # TODO Remove this method - it's obsolete if we use already divided data

        # train data preparation
        cut_all_into_segments(train_data_dir, Config.dataset_train_audio)

        noises = prepare_noise() if ADD_NOISE_TO_TRAINING_DATA else None

    nn_model = NNModel()

    if TRAINING_TYPE.train():
        train_ds, valid_ds = generate_train_valid_ds(noises)
        nn_model.train(train_ds, valid_ds)
    else:
        nn_model.load()

    if PREPARE_TEST_DATA:
        cut_all_into_segments(test_data_dir, Config.dataset_test)

    test_ds = generate_test_ds()

    predictions = nn_model.predict(test_ds)

    print_pretty_dict(predictions)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time} seconds")
