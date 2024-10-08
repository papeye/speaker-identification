import os
import numpy as np
import time

from data_preprocessing.audio_cutter import AudioCutter, AudiosCutter
from config import Config
from data_preprocessing.data_preparator import NoisePreparator
from data_preprocessing.dataset_generator import DatasetGenerator
from nnmodel import NNModel
from helpers import Helpers


def main():

    train_data_dir = "example_data/train_data"
    test_data_dir = "example_data/test_data"

    # base data preparation
    Helpers.move_base_data_to_proper_folders()  # TODO Remove this method - it's obsolety if we use already divided data

    # train data preparation
    AudiosCutter.cut_all_into_segments(train_data_dir, Config.dataset_train_audio)

    Helpers.resampleAll(Config.dataset_train_audio)

    # noise preparation
    noises = NoisePreparator().prepare()
    print("Noises moved to proper folders")

    ds_generator = DatasetGenerator()

    train_ds, valid_ds, class_names = ds_generator.generate_train_valid_ds(noises)

    # test data preparation
    AudiosCutter.cut_all_into_segments(test_data_dir, Config.dataset_test)
    Helpers.resampleAll(Config.dataset_test)

    # training
    nn_model = NNModel(len(class_names))
    nn_model.train(Config.epochs, train_ds, valid_ds)

    test_ds = ds_generator.generate_test_ds_from_paths()

    for audios, labels in test_ds.take(1):  # loop over batches
        y_pred = nn_model.predict(audios)
        y_pred = np.argmax(y_pred, axis=-1)

        predicted_label = class_names[y_pred[0]]
        print("Predicted Speaker:", predicted_label)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time} seconds")
