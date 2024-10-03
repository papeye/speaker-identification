import os


class Config:
    hugging_face_token = "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"
    dataset_root = "16000_pcm_speeches"
    dataset_train = "train_ds_dir"
    dataset_test = "test_ds_dir"

    audio_subfolder = "audio"
    noise_subfolder = "noise"

    dataset_audio_path = os.path.join(dataset_root, audio_subfolder)
    dataset_noise_path = os.path.join(dataset_root, noise_subfolder)
    valid_split = 0.1
    shuffle_seed = 43
    sampling_rate = 16000
    scale = 0.5
    batch_size = 128
    epochs = 1

    dataset_train_audio = os.path.join(dataset_train, audio_subfolder)
    dataset_train_noise = os.path.join(dataset_train, noise_subfolder)
    dataset_test_audio = os.path.join(dataset_test, audio_subfolder)
    dataset_test_noise = os.path.join(dataset_test, noise_subfolder)
