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

    # test_ds = ds_generator.generate_test_ds(noises)

    # SAMPLES_TO_DISPLAY = 50

    # for audios, labels in test_ds.take(1):
    #     # Get the signal FFT
    #     ffts = ds_generator.audio_to_fft(audios)
    #     # Predict
    #     y_pred = nn_model.predict(ffts)
    #     # Take random samples
    #     rnd = np.random.randint(0, Config.batch_size, SAMPLES_TO_DISPLAY)
    #     audios = audios.numpy()[rnd, :, :]
    #     labels = labels.numpy()[rnd]
    #     y_pred = np.argmax(y_pred, axis=-1)[rnd]

    #     for index in range(SAMPLES_TO_DISPLAY):
    #         # For every sample, print the true and predicted label
    #         # as well as run the voice with the noise
    #         # if labels[index] == 'ryczek':
    #         print(class_names[labels[index]], " ", class_names[y_pred[index]])
    #         #   print(
    #         #       "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
    #         #           "[92m" if labels[index] == y_pred[index] else "[91m",
    #         #           class_names[labels[index]],
    #         #           "[92m" if labels[index] == y_pred[index] else "[91m",
    #         #           class_names[y_pred[index]],
    #         #       )
    #         #   )
    #         # display(Audio(audios[index, :, :].squeeze(), rate=Config.sampling_rate))

    # Path to your test audio file
    test_audio_path = "test_data/edzik.wav"  # Replace with your actual path
    test_audio_paths = [test_audio_path]
    # Assuming 'speaker_name' is one of your class names
    # test_label = class_names.index("ryczekWav.wav")  # Replace with appropriate label
    # Print available class names
    print("Available class names:", class_names)

    # Replace 'ryczek' with the actual speaker name or use -1 if unknown
    speaker_name = "ryczekWav.wav"  # Replace with actual speaker name
    if speaker_name in class_names:
        test_label = class_names.index(speaker_name)
    else:
        test_label = -1  # Or handle as needed
        print(f"Speaker '{speaker_name}' not found in class names.")

    test_labels = [test_label]

    # Generate the test dataset from the new audio file
    test_ds = ds_generator.generate_test_ds_from_paths(
        noises, test_audio_paths, test_labels
    )

    for audios, labels in test_ds.take(1):
        print("audios shape:", audios.shape)
        # Get the signal FFT
        ffts = ds_generator.audio_to_fft(audios)
        print("ffts shape:", ffts.shape)

        # Ensure ffts has shape (batch_size, 8000, 1)
        if ffts.shape[1] != 8000:
            # Pad or trim ffts to have length 8000
            ffts = tf.pad(ffts, [[0, 0], [0, max(0, 8000 - ffts.shape[1])], [0, 0]])
            ffts = ffts[:, :8000, :]
            print("ffts shape after padding/trimming:", ffts.shape)

        # Predict
        y_pred = nn_model.predict(ffts)
        y_pred = np.argmax(y_pred, axis=-1)

        # Print the true and predicted labels
        actual_label = (
            class_names[labels.numpy()[0]] if labels.numpy()[0] != -1 else "Unknown"
        )
        predicted_label = class_names[y_pred[0]]
        print("Actual Speaker:", actual_label)
        print("Predicted Speaker:", predicted_label)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time} seconds")
