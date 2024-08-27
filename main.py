import os
from data_preprocessing.audio_cutter import AudioCutter
from config import Config
from data_preprocessing.data_preparator import DataPreparator

def main():
    audio_path = 'example_data/ryczekWav.wav'
    audio_name = os.path.basename(audio_path)
    output_path = os.path.join(Config.dataset_root, audio_name)
    new_audio_folder = os.path.join(Config.dataset_audio_path, audio_name)
    # AudioCutter(audio_path).cutAndAddToBaseData()
    DataPreparator(new_audio_folder).prepare()
    

if __name__ == "__main__":
    main()