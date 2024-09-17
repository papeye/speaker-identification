import os
from data_preprocessing.audio_cutter import AudioCutter
from config import Config
from data_preprocessing.data_preparator import DataPreparator
from data_preprocessing.dataset_generator import DatasetGenerator
from nnmodel import Model
import numpy as np
from IPython.display import display, Audio

def main():
    audio_path = 'example_data/ryczekWav.wav'
    audio_name = os.path.basename(audio_path)
    AudioCutter(audio_path).cutAndAddToBaseData()
    noises = DataPreparator().prepare(audio_name)
    train_ds, valid_ds, class_names,  valid_audio_paths, valid_labels =  DatasetGenerator().generate(noises)
    model1=Model(len(class_names))
    model1.train(1,train_ds,valid_ds)
    
    test_ds = DatasetGenerator().paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    
    SAMPLES_TO_DISPLAY = 10
    
    
    for audios, labels in test_ds.take(1):
        # Get the signal FFT
        ffts = DatasetGenerator().audio_to_fft(audios)
        # Predict
        y_pred = model1.predict(ffts)
        # Take random samples
        rnd = np.random.randint(0, Config.batch_size, SAMPLES_TO_DISPLAY)
        audios = audios.numpy()[rnd, :, :]
        labels = labels.numpy()[rnd]
        y_pred = np.argmax(y_pred, axis=-1)[rnd]

        for index in range(SAMPLES_TO_DISPLAY):
            # For every sample, print the true and predicted label
            # as well as run the voice with the noise
            if labels[index] == 'ryczek':
              print(
                  "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
                      "[92m" if labels[index] == y_pred[index] else "[91m",
                      class_names[labels[index]],
                      "[92m" if labels[index] == y_pred[index] else "[91m",
                      class_names[y_pred[index]],
                  )
              )
              display(Audio(audios[index, :, :].squeeze(), rate=Config.sampling_rate))
    
    

if __name__ == "__main__":
    main()