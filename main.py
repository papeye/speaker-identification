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

""" Flags for execution control"""
TRAINING_TYPE = TrainingType.PREPARE_DATA_AND_TRAIN
# TRAINING_TYPE = TrainingType.TRAIN_ONLY
# TRAINING_TYPE = TrainingType.NO_TRAINING

ADD_NOISE_TO_TRAINING_DATA = False
PREPARE_TEST_DATA = True


def main():
    train_data_dir = "example_data/train_data"
    test_data_dir = "example_data/test_data"

    if TRAINING_TYPE.prepareTrainData:
        start_time_prepare = time.time()
        move_base_data_to_proper_folders()  # TODO Remove this method - it's obsolete if we use already divided data

        cut_all_into_segments(train_data_dir, Config.dataset_train_audio)

        noises = prepareNoise() if ADD_NOISE_TO_TRAINING_DATA else None
        finish_time_prepare = time.time()



    nn_model = NNModel()

    if TRAINING_TYPE.train:
        start_time_training = time.time()
        train_ds, valid_ds = generate_train_valid_ds(noises)
        nn_model.train(train_ds, valid_ds)
        finish_time_training = time.time()
    else:
        nn_model.load()

    if PREPARE_TEST_DATA:
        start_time_test = time.time()
        for file in os.listdir(test_data_dir):
            path = os.path.join(test_data_dir, file)
            AudioCutter(path, Config.dataset_test).cut()
        finish_time_test = time.time()


    for dir in os.listdir(Config.dataset_test):
        start_time_predict = time.time()
        path = os.path.join(Config.dataset_test, dir)

        test_ds = generate_test_ds(path, dir)

        predictions = nn_model.predict(test_ds)
        finish_time_predict = time.time()

        print(f"Correct speaker: {dir}")
        printPrettyDict(predictions)
        
    #Below - times of execution
    if TRAINING_TYPE.prepareTrainData:
        print(f"Training data preparation took {finish_time_prepare - start_time_prepare} seconds")
    
    if TRAINING_TYPE.train:
        print(f"Training took {finish_time_training - start_time_training} seconds")
        
    if PREPARE_TEST_DATA:
        print(f"Test data preparation took {finish_time_test - start_time_test} seconds")
    
    print(f"Predictions took {finish_time_predict  - start_time_predict} seconds")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time} seconds")


