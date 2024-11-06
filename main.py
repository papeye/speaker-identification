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
# TRAINING_TYPE = TrainingType.PREPARE_DATA_AND_TRAIN
# TRAINING_TYPE = TrainingType.TRAIN_ONLY
TRAINING_TYPE = TrainingType.NO_TRAINING

ADD_NOISE_TO_TRAINING_DATA = False
PREPARE_TEST_DATA = False


def main():
    train_data_dir = "example_data/train_data"
    test_data_dir = "example_data/test_data"

    if TRAINING_TYPE.prepareTrainData:
        start_time_prepare = time.time()
        move_base_data_to_proper_folders()  # TODO Remove this method - it's obsolete if we use already divided data

        cut_all_into_segments(train_data_dir, Config.dataset_train_audio)

        noises = prepareNoise() if ADD_NOISE_TO_TRAINING_DATA else None
        time_prepare = time.time() - start_time_prepare



    nn_model = NNModel()

    if TRAINING_TYPE.train:
        start_time_training = time.time()
        train_ds, valid_ds = generate_train_valid_ds(noises)
        nn_model.train(train_ds, valid_ds)
        time_training = time.time() - start_time_training
    else:
        start_time_training = time.time()
        nn_model.load()
        time_training = time.time() - start_time_training


    if PREPARE_TEST_DATA:
        start_time_test = time.time()
        for file in os.listdir(test_data_dir):
            path = os.path.join(test_data_dir, file)
            AudioCutter(path, Config.dataset_test).cut()
        time_test = time.time() - start_time_test

    start_time_predict = time.time()
    for dir in os.listdir(Config.dataset_test):
        path = os.path.join(Config.dataset_test, dir)

        test_ds = generate_test_ds(path, dir)

        predictions = nn_model.predict(test_ds)
        
        print(f"Correct speaker: {dir}")
        printPrettyDict(predictions)
    time_predict = time.time() - start_time_predict

    print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    #Below - times of execution
    if TRAINING_TYPE.prepareTrainData:
        print(f"Training data preparation took {time_prepare} seconds")
    
    if TRAINING_TYPE.train:
        print(f"Training took {time_training} seconds")
    else:
        print(f"Model loading took {time_training} seconds")
        
    if PREPARE_TEST_DATA:
        print(f"Test data preparation took {time_test} seconds")
    
    print(f"Predictions took {time_predict} seconds")
    
    from datetime import datetime
    #Function for time 

    def get_last_modified_time(directory):
       # """Returns the last modified time of any file in the directory."""
        last_modified_time = 0  # Initialize to the earliest possible time
        last_modified_file = None 

        for root, _, files in os.walk(directory):
            for fname in files:
                file_path = os.path.join(root, fname)
                # Get the last modification time of the file
                mtime = os.path.getmtime(file_path)
                # Update last_modified_time if this file is newer
                if mtime > last_modified_time:
                    last_modified_time = mtime
                    last_modified_file = file_path


        # Convert the timestamp to a human-readable format
        if last_modified_time:
            return (
                last_modified_file,
                datetime.fromtimestamp(last_modified_time).strftime('%Y-%m-%d %H:%M:%S')
        )
        else:
            return "No files found."

# Usage example
    last_modified_default = get_last_modified_time(Config.dataset_root)
    last_modified_example = get_last_modified_time('example_data')
    print("Last modification time of any default file:", last_modified_default)
    print("Last modification time of any example file:", last_modified_example)



if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time} seconds")
