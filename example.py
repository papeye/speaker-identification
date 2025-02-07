from speaker_identifier import *

""" Flags for execution control"""
TRAINING_VAD = True  # whether to use VAD for training data
PREDICTING_VAD = True  # whether to use VAD for predicting data


def main():
    train_example_dir = "example_data/train_data"
    test_example_dir = "example_data/test_data"

    user1 = SpeakerIdentifier(
        model_name="user1",
        training_ds_dir=train_example_dir,
    )

    user1.train(
        train_data_dir=train_example_dir,
        with_vad=TRAINING_VAD,
    )

    predictions = user1.predict(
        test_data_dir=test_example_dir,
        with_vad=PREDICTING_VAD,
    )
    
    print("audio: prediction")
    for key, value in predictions.items():
        print(f"{key}: {value.best_prediction}")

    print(user1.timer)



if __name__ == "__main__":
    main()
