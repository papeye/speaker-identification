from speaker_identifier import SpeakerIdentifier, TrainingType, display_predictions


""" Flags for execution control"""
# TRAINING_TYPE = TrainingType.PREPARE_DATA_AND_TRAIN
TRAINING_TYPE = TrainingType.TRAIN_ONLY
# TRAINING_TYPE = TrainingType.NO_TRAINING

ADD_NOISE_TO_TRAINING_DATA = False
PREPARE_TEST_DATA = False

TRAINING_VAD = True
PREDICTING_VAD = True


def main():
    train_example_dir = "example_data/train_data"
    test_example_dir = "example_data/test_data"

    user1 = SpeakerIdentifier(model_name="user1")

    user1.train(
        train_data_dir=train_example_dir,
        training_type=TRAINING_TYPE,
        add_noise_to_training_data=ADD_NOISE_TO_TRAINING_DATA,
        with_vad=TRAINING_VAD,
    )

    predictions, correctly_identified = user1.predict(
        test_data_dir=test_example_dir,
        prepare_test_data=PREPARE_TEST_DATA,
        with_vad=PREDICTING_VAD,
    )

    display_predictions(predictions, correctly_identified)

    print(user1.timer)

    return correctly_identified


if __name__ == "__main__":
    main()
