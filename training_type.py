from enum import Enum


class TrainingType(Enum):
    PREPARE_DATA_AND_TRAIN = 1
    TRAIN_ONLY = 2
    NO_TRAINING = 3

    def prepareTrainData(self):
        return self == TrainingType.PREPARE_DATA_AND_TRAIN

    def train(self):
        return (
            self == TrainingType.PREPARE_DATA_AND_TRAIN
            or self == TrainingType.TRAIN_ONLY
        )
