import numpy as np
import time
from collections import Counter

from data_preprocessing.audio_cutter import AudiosCutter
from config import Config
from data_preprocessing.data_preparator import NoisePreparator
from data_preprocessing.dataset_generator import TrainDSGenerator, TestDSGenerator
from nnmodel import NNModel
from helpers import Helpers


def main():

    train_data_dir = "example_data/train_data"
    test_data_dir = "example_data/test_data"

    # base data preparation
    Helpers.move_base_data_to_proper_folders()  # TODO Remove this method - it's obsolety if we use already divided data

    # train data preparation
    AudiosCutter.cut_all_into_segments(train_data_dir, Config.dataset_train_audio)

    # noise preparation
    noises = NoisePreparator().prepare()
    print("Noises moved to proper folders")

    train_ds, valid_ds, class_names = TrainDSGenerator().generate_train_valid_ds(noises)

    # test data preparation
    AudiosCutter.cut_all_into_segments(test_data_dir, Config.dataset_test)

    # training
    nn_model = NNModel(len(class_names))
    nn_model.train(Config.epochs, train_ds, valid_ds)

    test_ds = TestDSGenerator().generate_test_ds()

    for audios, _ in test_ds:  # loop over batches
        y_pred = nn_model.predict(audios)
        y_pred = np.argmax(y_pred, axis=-1)

        predicted_labels = [class_names[i] for i in y_pred]

        for label, count in Counter(predicted_labels).items():
            print(f"{label} : {count} / {len(predicted_labels)}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time} seconds")
