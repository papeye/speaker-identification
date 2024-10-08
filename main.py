import os
import numpy as np
import time

from data_preprocessing.audio_cutter import AudioCutter
from config import Config
from data_preprocessing.data_preparator import DataPreparator
from data_preprocessing.dataset_generator import DatasetGenerator
from nnmodel import NNModel
import numpy as np
import tensorflow as tf

import h5py


def main():

    # Loop over train data
    for folder in os.listdir("train_data"):
        audio_path = os.path.join(
            "train_data", folder
        )  # We'll need to do this in the loop! For now it is what it is, basically loads audio for train and test and cuts it
        audio_name = os.path.basename(audio_path)
        output_path = Config.dataset_train_audio

        AudioCutter(audio_path, output_path).cutAndAddToBaseData()
        print("Audio cut and added to ", output_path)

        noises = DataPreparator().prepare(audio_name)
        print("Noises moved to proper folders")

    # Loop over test data
    for folder in os.listdir("test_data"):
        audio_path = os.path.join(
            "test_data", folder
        )  # We'll need to do this in the loop! For now it is what it is, basically loads audio for train and test and cuts it
        audio_name = os.path.basename(audio_path)
        output_path = Config.dataset_test_audio

        AudioCutter(audio_path, output_path).cutAndAddToBaseData()
        print("Audio cut and added to ", output_path)

    class_names = os.listdir(Config.dataset_train_audio)
    print(f"Found speakers: {class_names}")

    ds_generator = DatasetGenerator()

    train_ds, valid_ds = ds_generator.generate_train_valid_ds(noises, class_names)

    nn_model = NNModel(len(class_names))
    nn_model.train(Config.epochs, train_ds, valid_ds)

    # Path to your test audio file
    test_dir = "test_ds_dir/audio/edzik2.wav"  # C:\Work\python projects\speaker-identification\test_ds_dir\audio\edzik2.wav\1954 - 2954.wav

    file_paths = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, f))
    ]

    test_labels = [os.path.basename(os.path.dirname(file)) for file in file_paths]

    test_ds = ds_generator.generate_test_ds_from_paths(noises, file_paths, test_labels)

    for audios, labels in test_ds:

        # Predict
        y_pred = nn_model.predict(audios)
        # print(y_pred)
        y_pred = np.argmax(y_pred, axis=-1)

        for i in range(10):  # max is len(labels)
            predicted_label = class_names[y_pred[i]]
            print("Actual Speaker is", labels[i].numpy().decode("utf-8"))
            print("Predicted Speaker:", predicted_label)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time} seconds")
