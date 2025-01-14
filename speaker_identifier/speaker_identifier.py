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

        self.timer.start_prepare_train()
        self.speaker_labels = os.listdir(Config.dataset_train_audio)
        
        no_train_samples = len(os.listdir(train_data_dir))
        no_base_speakers = max(Config.n_speakers - no_train_samples, 0)

        move_base_data_to_proper_folders(no_base_speakers_to_move = no_base_speakers)
        cut_all_into_segments(
            train_data_dir, Config.dataset_train_audio, with_vad=with_vad,
        )

        self.timer.end_prepare_train()

        self.nn_model = NNModel(no_speakers = len(self.speaker_labels), model_name=self.name)

        self.timer.start_training()
        noises = prepareNoise() if add_noise_to_training_data else None

        train_ds, valid_ds = generate_train_valid_ds(noises)
        self.nn_model.train(train_ds, valid_ds)

        self.timer.end_training()

    def predict(
        self,
        test_data_dir: str,
        with_vad: bool = True,
    ) -> Result:
        self.timer.start_prepare_test()

        remove_dir(Config.dataset_test)

        for file in os.listdir(test_data_dir):
            path = os.path.join(test_data_dir, file)
            AudioCutter(path, Config.dataset_test).cut(with_vad)
            
        self.timer.end_prepare_test()

        result = Result()

        self.timer.start_predicting()

        for dir in os.listdir(Config.dataset_test):
            path = os.path.join(Config.dataset_test, dir)

            test_ds = generate_test_ds(path, dir)
            
            _predictions = self.nn_model.predict(test_ds)
            
            counts = Counter(_predictions)
            
            no_samples = len(_predictions)
            
            raw_result = {self.speaker_labels[i]: counts.get(i, 0) / no_samples for i in range(len(self.speaker_labels)) }
            
            result[dir] = dict(
                sorted((item for item in raw_result.items() if item[1] != 0), key=lambda item: item[1], reverse=True,)
            )
            
        self.timer.end_predict()
        self.timer.end_execution()

        return result
