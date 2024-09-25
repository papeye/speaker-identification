import os


class Config:
    hugging_face_token = "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"
    dataset_root = "16000_pcm_speeches"

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
    dataset_train = "train_ds_dir"
