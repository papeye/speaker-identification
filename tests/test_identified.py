from speaker_identifier import SpeakerIdentifier

minimun_threshold = 0.5  # TODO Update to 0.9 before release


def run_predicting() -> float:
    """
    Trains the speaker identifier model and tests its prediction accuracy.

    This function initializes the `SpeakerIdentifier` instance, trains the model
    using example training data, and evaluates its accuracy on example test data.

    Returns:
        float: The proportion of correctly identified speakers in the test data.
    """

    train_example_dir = "example_data/train_data"
    test_example_dir = "example_data/test_data"

    speaker_identifier = SpeakerIdentifier(training_ds_dir=train_example_dir)

    speaker_identifier.train(
        train_data_dir=train_example_dir,
    )

    _, correctly_identified = speaker_identifier.predict(
        test_data_dir=test_example_dir,
    )

    return correctly_identified


def test_correctly_identifed() -> None:
    """
    Test to ensure the model meets the minimum accuracy threshold.

    Asserts that the accuracy of the speaker identifier, as returned by
    `run_predicting`, is greater than or equal to the minimum threshold.

    Raises:
        AssertionError: If the accuracy is below the minimum threshold.
    """
    assert run_predicting() >= minimun_threshold
