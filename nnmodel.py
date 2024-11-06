import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "1"  # for This TensorFlow binary is optimized to use available CPU instructions...
)

import keras
import numpy as np
from collections import Counter

from config import Config


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
            patience=10, restore_best_weights=True
        )
        self.mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
            self.model_save_filename, monitor="val_accuracy", save_best_only=True
        )

    def __residual_block(self, x, filters, conv_num=3, activation="relu"):
        # Shortcut
        s = keras.layers.Conv1D(filters, 1, padding="same")(x)
        for i in range(conv_num - 1):
            x = keras.layers.Conv1D(filters, 3, padding="same")(x)
            x = keras.layers.Activation(activation)(x)
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Add()([x, s])
        x = keras.layers.Activation(activation)(x)
        return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)

    def __build_model(self, input_shape, num_classes):
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

    def train(self, train_ds, valid_ds):
        history = self.model.fit(
            train_ds,
            epochs=Config.epochs,
            validation_data=valid_ds,
            callbacks=[self.earlystopping_cb, self.mdlcheckpoint_cb],
        )

        self.model.save_weights("weights.weights.h5")

    def load(self):
        self.model.load_weights("weights.weights.h5")

    def predict(self, test_ds):

        audios, _ = next(iter(test_ds))

        # y_pred = self.model.predict(audios)
        y_pred = self.model(audios)

        y_pred_argmax = np.argmax(y_pred, axis=-1)

        predicted_labels = [self.speaker_labels[i] for i in y_pred_argmax]

        certainty_measure = 100 * np.mean(y_pred, axis=0)
        predicted_speaker_index = np.argmax(certainty_measure, axis=-1)
        predicted_speaker = self.speaker_labels[predicted_speaker_index]

        return predicted_speaker, certainty_measure, self.speaker_labels
