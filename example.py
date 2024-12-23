from speaker_identifier import *


""" Flags for execution control"""


def main():
    train_example_dir = "example_data/train_data"
    test_example_dir = "example_data/test_data"

    user1 = SpeakerIdentifier(model_name="user1")

    user1.train(
        train_data_dir=train_example_dir,
    )

    prediction_result = user1.predict(
        test_data_dir=test_example_dir,
    )

    print(prediction_result)
    print(user1.timer)


if __name__ == "__main__":
    main()
