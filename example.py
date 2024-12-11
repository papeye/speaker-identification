from speaker_identifier import SpeakerIdentifier, TrainingType, display_predictions


""" Flags for execution control"""
# TRAINING_TYPE = TrainingType.PREPARE_DATA_AND_TRAIN
TRAINING_TYPE = TrainingType.TRAIN_ONLY
# TRAINING_TYPE = TrainingType.NO_TRAINING

ADD_NOISE_TO_TRAINING_DATA = False
PREPARE_TEST_DATA = False


def main():
    train_example_dir = "example_data/train_data"
    test_example_dir = "example_data/test_data"

    user1 = SpeakerIdentifier(model_name="user1")

    user1.train(
        train_example_dir,
        TRAINING_TYPE,
        ADD_NOISE_TO_TRAINING_DATA,
    )

    predictions, correctly_identified = user1.predict(
        test_example_dir, PREPARE_TEST_DATA
    )

    display_predictions(predictions, correctly_identified)

    print(user1.timer)

    return correctly_identified


if __name__ == "__main__":
    main()
