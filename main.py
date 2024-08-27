import os
from data_preprocessing.audio_cutter import AudioCutter
from config import Config
from data_preprocessing.data_preparator import DataPreparator

def main():
    audio_path = 'example_data/ryczekWav.wav'
    audio_name = os.path.basename(audio_path)
    AudioCutter(audio_path).cutAndAddToBaseData()
    DataPreparator().prepare(audio_name)
    

if __name__ == "__main__":
    main()