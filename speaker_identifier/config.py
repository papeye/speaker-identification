import os

package_dir = os.path.dirname(__file__)


class Config:
    dataset_root = os.path.join(package_dir, "16000_pcm_speeches")

    dataset_train = "train_ds_dir"
    dataset_test = "test_ds_dir"

    n_speakers = 20

    audio_subfolder = "audio"
    noise_subfolder = "noise"

    dataset_audio_path = os.path.join(dataset_root, audio_subfolder)
    dataset_noise_path = os.path.join(dataset_root, noise_subfolder)

    dataset_train_audio = os.path.join(dataset_train, audio_subfolder)
    dataset_train_noise = os.path.join(dataset_train, noise_subfolder)

    valid_split = 0.1
    shuffle_seed = 43
    sampling_rate = 16000
    sample_width = 2  # 16-bit audio uses 2 bytes per sample (since 16 bits = 2 bytes)
    scale = 0.5
    batch_size = 128
    epochs = 3


class Utils:
    @staticmethod
    def model_file_path(model_filename: str) -> str:
        return os.path.join(package_dir, model_filename)
