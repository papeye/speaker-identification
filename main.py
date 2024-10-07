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

    # # base data preparation
    Helpers.move_base_data_to_proper_folders()  # TODO Remove this method - it's obsolety if we use already divided data

    # train data preparation
    AudiosCutter.cut_all_into_segments(train_data_dir, Config.dataset_train_audio)

    Helpers.resampleAll(Config.dataset_train_audio)

    # noise preparation
    noises = NoisePreparator().prepare()
    print("Noises moved to proper folders")

    # class names
    class_names = os.listdir(Config.dataset_train_audio)
    print(f"Found speakers: {class_names}")

    ds_generator = DatasetGenerator()

    train_ds, valid_ds = ds_generator.generate_train_valid_ds(noises, class_names)

    # test data preparation
    AudiosCutter.cut_all_into_segments(test_data_dir, Config.dataset_test)
    Helpers.resampleAll(Config.dataset_test)

    # training
    nn_model = NNModel(len(class_names))
    nn_model.train(Config.epochs, train_ds, valid_ds)

    all = 0
    matches = 0

    speakers_match = []

    for dir in os.listdir(Config.dataset_test):
        dir_path = os.path.join(Config.dataset_test, dir)

        match_speaker = 0
        all_samples = 0

        for file in os.listdir(dir_path):
            test_audio_path = os.path.join(dir_path, file)

            if os.path.isdir(test_audio_path):
                raise RuntimeError(
                    f"{test_audio_path} is a directory, not aa audio file"
                )

            test_labels = [dir]

            # Generate the test dataset from the new audio file
            test_ds = DatasetGenerator().generate_test_ds_from_paths(
                [test_audio_path], test_labels
            )  # I think there is no need for loop here

            for audios, labels in test_ds.take(1):
                # Predict
                y_pred = nn_model.predict(audios)
                y_pred = np.argmax(y_pred, axis=-1)

                # Print the true and predicted labels

                predicted_label = class_names[y_pred[0]]

                all += 1
                all_samples += 1
                if predicted_label == dir:
                    matches += 1
                    match_speaker += 1

        speakers_match.append((dir, match_speaker, all_samples))
        print(f"Matched {match_speaker} / {all_samples} samples for speaker {dir}")

    for speaker, match_speaker, all_samples in speakers_match:
        print(f"Matched {match_speaker} / {all_samples} samples for speaker {speaker}")
    print(f"Totally matched {matches} / {all} samples")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time} seconds")
