from speaker_identifier.training_type import TrainingType
from speaker_identifier.speaker_identifier import SpeakerIdentifier


def run_main():
    train_example_dir = "example_data/train_data"
    test_example_dir = "example_data/test_data"

    speaker_identifier = SpeakerIdentifier()

    speaker_identifier.train(
        train_data_dir=train_example_dir,
        training_type=TrainingType.PREPARE_DATA_AND_TRAIN,
        add_noise_to_training_data=False,
    )

    _, correctly_identified = speaker_identifier.predict(
        test_data_dir=test_example_dir,
        prepareTestData=True,
    )

    return correctly_identified


def test_correctly_identifed():
    # TODO Once we fix testing data this number needs to be increased to at least 0.9
    assert run_main() >= 0.5
