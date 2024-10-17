import numpy as np
import time
from collections import Counter

from data_preprocessing.audio_cutter import AudiosCutter
from config import Config
from data_preprocessing.data_preparator import NoisePreparator
from data_preprocessing.dataset_generator import TrainDSGenerator, TestDSGenerator
from nnmodel import NNModel
from helpers import Helpers
from training_type import TrainingType

""" Flags for execution control"""
TRAINING_TYPE = TrainingType.PREPARE_DATA_AND_TRAIN
# TRAINING_TYPE = TrainingType.TRAIN_ONLY
# TRAINING_TYPE = TrainingType.NO_TRAINING

ADD_NOISE_TO_TRAINING_DATA = False
PREPARE_TEST_DATA = True


def main():
    train_data_dir = "example_data/train_data"
    test_data_dir = "example_data/test_data"

    if TRAINING_TYPE.prepareTrainData():
        Helpers.move_base_data_to_proper_folders()  # TODO Remove this method - it's obsolete if we use already divided data

        # train data preparation
        AudiosCutter.cut_all_into_segments(train_data_dir, Config.dataset_train_audio)

        noises = NoisePreparator().prepare() if ADD_NOISE_TO_TRAINING_DATA else None

    nn_model = NNModel()

    if TRAINING_TYPE.train():  # if we prepare data, we need to train model
        train_ds, valid_ds = TrainDSGenerator().generate_train_valid_ds(noises)
        nn_model.train(train_ds, valid_ds)
    else:
        nn_model.load()

    if PREPARE_TEST_DATA:
        AudiosCutter.cut_all_into_segments(test_data_dir, Config.dataset_test)

    test_ds = TestDSGenerator().generate_test_ds()

    predictions = nn_model.predict(test_ds)

    Helpers.printPrettyDict(predictions)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time} seconds")
