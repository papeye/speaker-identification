import os
import keras
import numpy as np
import tensorflow as tf
import math


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "1"  # for This TensorFlow binary is optimized to use available CPU instructions...
)

from .config import Config, Utils


class NNModel:

    def __compile_model(self):
        self.model.compile(
            optimizer="Adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def __init__(self, model_name: str):
        _model_filename = (
            f"{model_name}.keras"
            if model_name != None
            else f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.keras"
        )
        self.model_filepath = Utils.model_file_path(_model_filename)

        self.speaker_labels = os.listdir(Config.dataset_train_audio)
        self.num_classes = len(self.speaker_labels)

        try:
            self.model = keras.models.load_model(self.model_filepath)
        except ValueError as e:
            print("Model file not found, creating new model")
            self.__build_model((Config.sampling_rate // 2, 1))

        self.earlystopping_cb = keras.callbacks.EarlyStopping(
            patience=2, restore_best_weights=True
        )

        self.early_stopping_accuracy = EarlyStoppingByAccuracy(
            target_accuracy=Config.target_accuracy
        )

        self.mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
            self.model_filepath, monitor="val_accuracy", save_best_only=True
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

    def __build_model(self, input_shape) -> keras.Model:
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

        outputs = keras.layers.Dense(
            self.num_classes, activation="softmax", name="output"
        )(x)

        self.model = keras.models.Model(inputs=inputs, outputs=outputs)
        self.__compile_model()

    def _update_output_layer(self):
        print(
            f"Replacing the output layer with a new one with {self.num_classes} classes"
        )

        x = self.model.layers[-2].output

        new_output = keras.layers.Dense(
            self.num_classes, activation="softmax", name="output"
        )(x)

        self.model = keras.models.Model(inputs=self.model.input, outputs=new_output)

        self.__compile_model()

    def train(self, train_ds: tf.Tensor, valid_ds: tf.Tensor) -> None:
        self._update_output_layer()
        steps_per_epoch = math.ceil(train_ds.cardinality().numpy() * 0.6)
        print(steps_per_epoch)

        self.history = self.model.fit(
            train_ds.repeat(),
            epochs=Config.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_ds,
            callbacks=[
                self.earlystopping_cb,
                self.mdlcheckpoint_cb,
                self.early_stopping_accuracy,
            ],
        )

        print(f"Training finished on model: \n {self.model.summary()}")

    def predict(self, test_ds: tf.data.Dataset) -> dict[str, float]:
        audios, _ = next(iter(test_ds))

        y_pred = self.model(audios)

        certainty_measure = 100 * np.mean(y_pred, axis=0)
        predicted_speaker_index = np.argmax(certainty_measure, axis=-1)
        predicted_speaker = self.speaker_labels[predicted_speaker_index]

        return predicted_speaker, certainty_measure, self.speaker_labels


class EarlyStoppingByAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy):
        super().__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get("val_accuracy")
        if val_accuracy and val_accuracy >= self.target_accuracy:
            print(
                f"\nStopping training early: val_accuracy reached {val_accuracy:.4f} at epoch {epoch + 1}"
            )
            self.model.stop_training = True
