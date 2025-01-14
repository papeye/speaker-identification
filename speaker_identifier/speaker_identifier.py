import os
import datetime
from collections import Counter

from .timer import Timer
from .config import Config
from .nnmodel import NNModel
from .helpers import move_base_data_to_proper_folders, remove_dir
from .data_preprocessing.audio_cutter import AudioCutter, cut_all_into_segments
from .data_preprocessing.noise_preparator import prepareNoise
from .data_preprocessing.dataset_generator import (
    generate_train_valid_ds,
    generate_test_ds,
)
from .result import Result


class SpeakerIdentifier:
    def __init__(self, training_ds_dir: str, model_name: str = "<timestamp>") -> None:
        self.timer = Timer()

        self.training_ds_dir = training_ds_dir
        self.name = (
            model_name
            if model_name != "<timestamp>"
            else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

    def train(
        self,
        train_data_dir: str,
        add_noise_to_training_data: bool = Config.default_add_noise_to_train_data,
        with_vad: bool = True,
    ) -> None:
        """
        Train the neural network model on the provided training data.

        This method prepares the training dataset by segmenting audio files into smaller segments,
        optionally adding noise for data augmentation, and splits the data into training and validation sets.
        The neural network model is then trained using the prepared datasets.

        Args:
            train_data_dir (str): Path to the directory containing training audio files.
            add_noise_to_training_data (bool, optional): Whether to add noise to training data for data augmentation.
                                                         Defaults to `Config.default_add_noise_to_train_data`.
            with_vad (bool, optional): Whether to apply Voice Activity Detection (VAD) during preprocessing.
                                      Defaults to `True`.

        Returns:
            None: The method trains the model and doesn't return a value.

        Workflow:
            1. Prepares the training dataset by segmenting audio files (with optional VAD).
            2. Moves base data into proper folders based on speaker labels.
            3. Optionally adds noise to training data for augmentation.
            4. Splits the data into training and validation sets.
            5. Trains the neural network model using the prepared datasets.

        Notes:
            - This method relies on several helper functions like `move_base_data_to_proper_folders`,
              `cut_all_into_segments`, and `generate_train_valid_ds` for preprocessing and dataset generation.
            - The neural network model is instantiated with the number of speakers determined by the
              length of `self.speaker_labels`.
            - The `self.timer` is used to track different stages of training, such as dataset preparation,
              training, and total execution time.

        Example:
            >>> model.train("/path/to/train/data", add_noise_to_training_data=True, with_vad=True)
        """

        self.timer.start_prepare_train()

        no_train_samples = len(os.listdir(train_data_dir))
        no_base_speakers = max(Config.n_speakers - no_train_samples, 0)

        move_base_data_to_proper_folders(no_base_speakers_to_move=no_base_speakers)
        cut_all_into_segments(
            train_data_dir,
            Config.dataset_train_audio,
            with_vad=with_vad,
        )

        self.speaker_labels = os.listdir(Config.dataset_train_audio)

        self.timer.end_prepare_train()

        self.nn_model = NNModel(
            no_speakers=len(self.speaker_labels), model_name=self.name
        )

        self.timer.start_training()
        noises = prepareNoise() if add_noise_to_training_data else None

        train_ds, valid_ds = generate_train_valid_ds(noises)
        self.nn_model.train(train_ds, valid_ds)

        self.timer.end_training()

    def predict(
        self,
        test_data_dir: str,
        with_vad: bool = True,
    ) -> dict[str, Result]:
        """
        Predict speaker labels for audio samples in the specified directory.

        This method processes audio files located in the specified directory, optionally applies
        Voice Activity Detection (VAD) during preprocessing, and generates predictions for each
        speaker. The results are sorted by probability and returned as a dictionary.

        Args:
            test_data_dir (str): Path to the directory containing test audio files. Each file is
                                 assumed to belong to one speaker.
            with_vad (bool, optional): Whether to apply Voice Activity Detection (VAD) while cutting
                                       audio samples. Defaults to `True`.

        Returns:
            Result: A structured result object mapping each speaker directory to a dictionary
                    of predicted speaker labels and their normalized probabilities, sorted in
                    descending order of probability.

        Workflow:
            1. Prepares the test dataset by cutting audio files (with optional VAD).
            2. Generates predictions for each speaker directory using the neural network model.
            3. Calculates the normalized probabilities for each speaker label.
            4. Returns the predictions sorted by probability for each directory.

        Notes:
            - This method relies on `AudioCutter` for preprocessing and `generate_test_ds`
              for creating TensorFlow datasets.
            - Predictions are made using the neural network model stored in `self.nn_model`.
            - The `self.timer` tracks execution time for different stages (e.g., dataset preparation,
              prediction, and total execution).

        Example:
            >>> result = model.predict("/path/to/test/data", with_vad=True)
            >>> print(result["speaker1"])
            {
                "Speaker A": 0.75,
                "Speaker B": 0.25
            }
        """
        self.timer.start_prepare_test()

        remove_dir(Config.dataset_test)

        for file in os.listdir(test_data_dir):
            path = os.path.join(test_data_dir, file)
            AudioCutter(path, Config.dataset_test).cut(with_vad)

        self.timer.end_prepare_test()

        result = {}

        self.timer.start_predicting()

        for dir in os.listdir(Config.dataset_test):
            path = os.path.join(Config.dataset_test, dir)

            test_ds = generate_test_ds(path, dir)

            _predictions = self.nn_model.predict(test_ds)

            counts = Counter(_predictions)

            no_samples = len(_predictions)

            raw_result = {
                self.speaker_labels[i]: counts.get(i, 0) / no_samples
                for i in range(len(self.speaker_labels))
            }

            result[dir] = Result(
                sorted(
                    (item for item in raw_result.items() if item[1] != 0),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

        self.timer.end_predict()
        self.timer.end_execution()

        return result
