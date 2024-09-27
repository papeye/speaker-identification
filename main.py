import os
import numpy as np
import time

from data_preprocessing.audio_cutter import AudioCutter
from config import Config
from data_preprocessing.data_preparator import DataPreparator
from data_preprocessing.dataset_generator import DatasetGenerator
from nnmodel import NNModel



def main():
    audio_path = "example_data/ryczekWav.wav"
    audio_name = os.path.basename(audio_path)

    AudioCutter(audio_path).cutAndAddToBaseData()
    print("Audio cut and added to ", Config.dataset_train_audio)
    

    noises = DataPreparator().prepare(audio_name)
    print("Noises moved to proper folders")

    class_names = os.listdir(Config.dataset_train_audio)
    print(f"Found speakers: {class_names}")

    ds_generator = DatasetGenerator()

    train_ds, valid_ds = ds_generator.generate_train_valid_ds(noises, class_names)

    nn_model = NNModel(len(class_names))
    nn_model.train(Config.epochs, train_ds, valid_ds)

    test_ds = ds_generator.generate_test_ds(noises)

    SAMPLES_TO_DISPLAY = 50

    for audios, labels in test_ds.take(1):
        # Get the signal FFT
        ffts = ds_generator.audio_to_fft(audios)
        # Predict
        y_pred = nn_model.predict(ffts)
        # Take random samples
        rnd = np.random.randint(0, Config.batch_size, SAMPLES_TO_DISPLAY)
        audios = audios.numpy()[rnd, :, :]
        labels = labels.numpy()[rnd]
        y_pred = np.argmax(y_pred, axis=-1)[rnd]

        for index in range(SAMPLES_TO_DISPLAY):
            # For every sample, print the true and predicted label
            # as well as run the voice with the noise
            # if labels[index] == 'ryczek':
            print(class_names[labels[index]], " ", class_names[y_pred[index]])
            #   print(
            #       "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
            #           "[92m" if labels[index] == y_pred[index] else "[91m",
            #           class_names[labels[index]],
            #           "[92m" if labels[index] == y_pred[index] else "[91m",
            #           class_names[y_pred[index]],
            #       )
            #   )
            # display(Audio(audios[index, :, :].squeeze(), rate=Config.sampling_rate))


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution took {time.time() - start_time} seconds")
