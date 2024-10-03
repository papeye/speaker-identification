import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np

import tensorflow as tf

from pathlib import Path

from config import Config


class DatasetGenerator:

    def __path_to_audio(self, path):
        """Reads and decodes an audio file."""
        audio = tf.io.read_file(path)
        audio, _ = tf.audio.decode_wav(audio, 1, Config.sampling_rate)
        return audio

    def __paths_and_labels_to_dataset(self, audio_paths, labels):
        """Constructs a dataset of audios and labels."""
        path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = path_ds.map(
            lambda x: self.__path_to_audio(x), num_parallel_calls=tf.data.AUTOTUNE
        )
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        return tf.data.Dataset.zip((audio_ds, label_ds))

    def __add_noise(self, audio, noises=None, scale=0.5):
        if noises is not None:
            # Create a random tensor of the same size as audio ranging from
            # 0 to the number of noise stream samples that we have.
            tf_rnd = tf.random.uniform(
                (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
            )
            noise = tf.gather(noises, tf_rnd, axis=0)

            # Get the amplitude proportion between the audio and the noise
            prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
            prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

            # Adding the rescaled noise to audio
            audio = audio + noise * prop * scale

        return audio

    def audio_to_fft(self, audio):
        # Since tf.signal.fft applies FFT on the innermost dimension,
        # we need to squeeze the dimensions and then expand them again
        # after FFT
        audio = tf.squeeze(audio, axis=-1)
        fft = tf.signal.fft(
            tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
        )
        fft = tf.expand_dims(fft, axis=-1)

        # Return the absolute value of the first half of the FFT
        # which represents the positive frequencies
        return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

    # Get the list of audio file paths along with their corresponding labels

    def generate_train_valid_ds(self, noises, class_names):

        audio_paths = []
        labels = []
        for label, name in enumerate(class_names):
            print(
                "Processing speaker {}".format(
                    name,
                )
            )
            dir_path = Path(Config.dataset_train_audio) / name
            speaker_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            audio_paths += speaker_sample_paths
            labels += [label] * len(speaker_sample_paths)

        print(
            "Found {} files belonging to {} classes.".format(
                len(audio_paths), len(class_names)
            )
        )

        # Shuffle
        rng = np.random.RandomState(Config.shuffle_seed)
        rng.shuffle(audio_paths)
        rng = np.random.RandomState(Config.shuffle_seed)
        rng.shuffle(labels)

        # Split into training and validation
        num_val_samples = int(Config.valid_split * len(audio_paths))
        print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
        train_audio_paths = audio_paths[:-num_val_samples]
        train_labels = labels[:-num_val_samples]

        print("Using {} files for validation.".format(num_val_samples))
        valid_audio_paths = audio_paths[-num_val_samples:]
        valid_labels = labels[-num_val_samples:]

        # Create 2 datasets, one for training and the other for validation
        train_ds = self.__paths_and_labels_to_dataset(train_audio_paths, train_labels)
        train_ds = train_ds.shuffle(
            buffer_size=Config.batch_size * 8, seed=Config.shuffle_seed
        ).batch(Config.batch_size)

        valid_ds = self.__paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
        self.valid_ds = valid_ds

        valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=Config.shuffle_seed).batch(
            32
        )

        # Add noise to the training set
        train_ds = train_ds.map(
            lambda x, y: (self.__add_noise(x, noises, scale=Config.scale), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Transform audio wave to the frequency domain using `audio_to_fft`
        train_ds = train_ds.map(
            lambda x, y: (self.audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        valid_ds = valid_ds.map(
            lambda x, y: (self.audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

        self.valid_audio_paths = valid_audio_paths
        self.valid_labels = valid_labels

        return train_ds, valid_ds

    def generate_test_ds(self, noises):
        test_ds = self.valid_ds.shuffle(
            buffer_size=Config.batch_size * 8, seed=Config.shuffle_seed
        ).batch(Config.batch_size)
        test_ds = test_ds.map(
            lambda x, y: (self.__add_noise(x, noises, scale=Config.scale), y),
            # num_parallel_calls=tf.data.AUTOTUNE,
        )

        return test_ds
