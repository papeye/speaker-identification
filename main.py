import os
from data_preprocessing.audio_cutter import AudioCutter
from config import Config
from data_preprocessing.data_preparator import DataPreparator
from data_preprocessing.dataset_generator import DatasetGenerator
from nnmodel import Model
import numpy as np
from IPython.display import display, Audio
import tensorflow as tf

def main():
    audio_path = 'example_data/ryczekWav.wav'
    audio_name = os.path.basename(audio_path)
    AudioCutter(audio_path).cutAndAddToBaseData()
    noises = DataPreparator().prepare(audio_name)
    train_ds, valid_ds, class_names,  valid_audio_paths, valid_labels =  DatasetGenerator().generate(noises)
    print(111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111)
    model1=Model(len(class_names))
    model1.train(1,train_ds,valid_ds)
    print(2222222222222222222222222222222222222222222222222222222222222222222222222222222)
    test_ds = DatasetGenerator().paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    test_ds = test_ds.shuffle(buffer_size=Config.batch_size * 8, seed=Config.shuffle_seed).batch(Config.batch_size)
    print(3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333)
    test_ds = test_ds.map(
        lambda x, y: (DatasetGenerator().add_noise(x, noises, scale=Config.scale), y),
        #num_parallel_calls=tf.data.AUTOTUNE,
    )
    
    SAMPLES_TO_DISPLAY = 50
    print(444444444444444444444444444444444444444444444)
    for audios, labels in test_ds.take(1):
        # Get the signal FFT
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        ffts = DatasetGenerator().audio_to_fft(audios)
        print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        # Predict
        y_pred = model1.predict(ffts)
        print('cccccccccccccccccccccccccccccccccccc')
        # Take random samples
        rnd = np.random.randint(0, Config.batch_size, SAMPLES_TO_DISPLAY)
        audios = audios.numpy()[rnd, :, :]
        labels = labels.numpy()[rnd]
        y_pred = np.argmax(y_pred, axis=-1)[rnd]

        print(55555555555555555555555555555555)

        for index in range(SAMPLES_TO_DISPLAY):
            # For every sample, print the true and predicted label
            # as well as run the voice with the noise
            #if labels[index] == 'ryczek':
            print(class_names[labels[index]], ' ', class_names[y_pred[index]])
            #   print(
            #       "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
            #           "[92m" if labels[index] == y_pred[index] else "[91m",
            #           class_names[labels[index]],
            #           "[92m" if labels[index] == y_pred[index] else "[91m",
            #           class_names[y_pred[index]],
            #       )
            #   )
              #display(Audio(audios[index, :, :].squeeze(), rate=Config.sampling_rate))
    
    

if __name__ == "__main__":
    main()