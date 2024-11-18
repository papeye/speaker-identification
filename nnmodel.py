import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "1"  # for This TensorFlow binary is optimized to use available CPU instructions...
)

import keras
import numpy as np
import tensorflow as tf
from config import Config
import h5py


class NNModel:
    def __init__(self):
        self.speaker_labels = os.listdir(Config.dataset_train_audio)
        self.model = self.__build_model(
            (Config.sampling_rate // 2, 1), len(self.speaker_labels)
        )
        self.model.compile(
            optimizer="Adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model_save_filename = "model.keras"
        self.earlystopping_cb = keras.callbacks.EarlyStopping(
            patience=2, restore_best_weights=True
        )
        self.mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
            self.model_save_filename, monitor="val_accuracy", save_best_only=True
        )

    def __residual_block(
        self, x: tf.Tensor, filters: int, conv_num=3, activation="relu"
    ) -> tf.Tensor:
        # Shortcut
        s = keras.layers.Conv1D(filters, 1, padding="same")(x)
        for i in range(conv_num - 1):
            x = keras.layers.Conv1D(filters, 3, padding="same")(x)
            x = keras.layers.Activation(activation)(x)
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Add()([x, s])
        x = keras.layers.Activation(activation)(x)
        return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)

    def __build_model(self, input_shape, num_classes: int) -> keras.Model:
        inputs = keras.layers.Input(shape=input_shape, name="input")

        x = self.__residual_block(inputs, 16, 2)
        x = self.__residual_block(x, 32, 2)
        x = self.__residual_block(x, 64, 3)
        x = self.__residual_block(x, 128, 3)
        x = self.__residual_block(x, 128, 3)

        x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(128, activation="relu")(x)

        outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(
            x
        )

        return keras.models.Model(inputs=inputs, outputs=outputs)

    def train(self, train_ds: tf.Tensor, valid_ds: tf.Tensor) -> None:
        self.load()
        history = self.model.fit(
            train_ds,
            epochs=Config.epochs,
            validation_data=valid_ds,
            callbacks=[self.earlystopping_cb, self.mdlcheckpoint_cb],
        )

        self.model.save_weights("weights.weights.h5")

    def load(self) -> None:
        output_dim = get_output_dim_from_weights("weights.weights.h5", "dense_2")
        if output_dim == len(os.listdir(Config.dataset_train_audio)):
            print("The number of speakers didn't change so previous weights are loaded")
            self.model.load_weights("weights.weights.h5")
        else:
            print("No model was loaded")

    def predict(self, test_ds: tf.data.Dataset) -> dict[str, float]:
        audios, _ = next(iter(test_ds))
        y_pred = self.model(audios)
        certainty_measure = 100 * np.mean(y_pred, axis=0)
        predicted_speaker_index = np.argmax(certainty_measure, axis=-1)
        predicted_speaker = self.speaker_labels[predicted_speaker_index]

        return predicted_speaker, certainty_measure, self.speaker_labels


def get_output_dim_from_weights(weights_file, output_layer_name):
    with h5py.File(weights_file, "r") as f:
        layers_group = f["layers"]
        if output_layer_name in layers_group:
            layer = layers_group[output_layer_name]
            if "vars" in layer:
                vars_group = layer["vars"]
                # Identify weights dataset (the one with 2D shape)
                for var_name in vars_group.keys():
                    var_data = vars_group[var_name]
                    if len(var_data.shape) == 2:
                        weights = var_data
                        output_dim = weights.shape[1]
                        return output_dim
    return None
