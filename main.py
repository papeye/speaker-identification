import os
from data_preprocessing.audio_cutter import AudioCutter
from config import Config
from data_preprocessing.data_preparator import DataPreparator
from data_preprocessing.dataset_generator import DatasetGenerator

def main():
    audio_path = 'example_data/ryczekWav.wav'
    audio_name = os.path.basename(audio_path)
    AudioCutter(audio_path).cutAndAddToBaseData()
    noises = DataPreparator().prepare(audio_name)
    train_ds, valid_ds =  DatasetGenerator().generate(noises)
    
    

if __name__ == "__main__":
    main()