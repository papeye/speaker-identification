from timer import Timer
import os


from config import Config
from helpers import move_base_data_to_proper_folders, remove_dir
from data_preprocessing.audio_cutter import AudioCutter, cut_all_into_segments
from data_preprocessing.noise_preparator import prepareNoise
from data_preprocessing.dataset_generator import (
    generate_train_valid_ds,
    generate_test_ds,
)
from nnmodel import NNModel
import numpy as np
from training_type import TrainingType


class SpeakerIdentifier:
    def __init__(self):
        self.timer = Timer()
        self.timer.start_executing()
        self.nn_model = NNModel()

    def train(
        self,
        train_data_dir: dir,
        training_type: TrainingType,
        add_noise_to_training_data: bool,
    ) -> None:
        if training_type.prepareTrainData:
            self.timer.start_prepare_train()
            move_base_data_to_proper_folders()  # TODO Remove this method - it's obsolete if we use already divided data
            cut_all_into_segments(train_data_dir, Config.dataset_train_audio)
            self.noises = prepareNoise() if add_noise_to_training_data else None
            self.timer.end_prepare_train()
        if training_type.train:
            self.timer.start_training()
            train_ds, valid_ds = generate_train_valid_ds(self.noises)
            self.nn_model.train(train_ds, valid_ds)
            self.timer.end_training()
        else:
            self.nn_model.load()

    def predict(self, test_data_dir: str, prepareTestData: bool) -> None:
        if prepareTestData:
            self.timer.start_prepare_test()

            remove_dir(Config.dataset_test)

            for file in os.listdir(test_data_dir):
                path = os.path.join(test_data_dir, file)
                AudioCutter(path, Config.dataset_test).cut()
            self.timer.end_prepare_test()

        correctly_identyfied = 0

        self.timer.start_predicting()
        for dir in os.listdir(Config.dataset_test):
            path = os.path.join(Config.dataset_test, dir)

            test_ds = generate_test_ds(path, dir)

            predicted_speaker, certainty_measure, speaker_labels = (
                self.nn_model.predict(test_ds)
            )
            if predicted_speaker == dir:
                correctly_identyfied += 1

            print(
                f"\n Correct speaker: {dir}, predicted speaker is {predicted_speaker}"
            )

            max_prediction = np.max(certainty_measure)

            for i in range(len(certainty_measure)):
                if certainty_measure[i] > 5:
                    if (
                        certainty_measure[i] == max_prediction
                        and speaker_labels[i] == dir
                    ):
                        print(
                            f"\033[1;32;40m {speaker_labels[i]}: {certainty_measure[i]:.2f}% \033[0m"
                        )
                    elif (
                        certainty_measure[i] == max_prediction
                        and speaker_labels[i] != dir
                    ):
                        print(
                            f"\033[1;31;40m {speaker_labels[i]}: {certainty_measure[i]:.2f} %\033[0m"
                        )
                    else:
                        print(f"{speaker_labels[i]}: {certainty_measure[i]:.2f}%")

        print(
            f"\n Correctly identified speakers: {correctly_identyfied} out of {len(os.listdir(Config.dataset_test))}"
        )
        print(f"Correct speaker: {dir}")

        self.timer.end_predict()

        self.timer.end_execution()
