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

    def train(
        self,
        train_data_dir: dir,
        training_type: TrainingType,
        add_noise_to_training_data: bool,
    ) -> None:
        self.nn_model = NNModel()
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

        correctly_identified = 0
        predictions = []

        self.timer.start_predicting()

        for dir in os.listdir(Config.dataset_test):
            path = os.path.join(Config.dataset_test, dir)

            test_ds = generate_test_ds(path, dir)

            predicted_speaker, certainty_measure, speaker_labels = (
                self.nn_model.predict(test_ds)
            )
            if predicted_speaker == dir:
                correctly_identified += 1

            predictions.append(
                {
                    "correct_speaker": dir,
                    "predicted_speaker": predicted_speaker,
                    "certainty_measure": certainty_measure,
                    "speaker_labels": speaker_labels,
                }
            )

        self.timer.end_predict()
        self.timer.end_execution()

        return predictions, correctly_identified

    def display_predictions(self, predictions, correctly_identified):
        total_speakers = len(os.listdir(Config.dataset_test))

        for detail in predictions:
            correct_speaker = detail["correct_speaker"]
            predicted_speaker = detail["predicted_speaker"]
            certainty_measure = detail["certainty_measure"]
            speaker_labels = detail["speaker_labels"]
            max_prediction = np.max(certainty_measure)

            print(
                f"\nCorrect speaker: {correct_speaker}, predicted speaker is {predicted_speaker}"
            )

            for i in range(len(certainty_measure)):
                if certainty_measure[i] > 5:
                    if (
                        certainty_measure[i] == max_prediction
                        and speaker_labels[i] == correct_speaker
                    ):
                        print(
                            f"\033[1;32;40m {speaker_labels[i]}: {certainty_measure[i]:.2f}% \033[0m"
                        )
                    elif (
                        certainty_measure[i] == max_prediction
                        and speaker_labels[i] != correct_speaker
                    ):
                        print(
                            f"\033[1;31;40m {speaker_labels[i]}: {certainty_measure[i]:.2f}% \033[0m"
                        )
                    else:
                        print(f"{speaker_labels[i]}: {certainty_measure[i]:.2f}%")

        print(
            f"\nCorrectly identified speakers: {correctly_identified} out of {total_speakers}"
        )
