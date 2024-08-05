from helpers import Helpers
from data_preprocessing.audio_cutter import AudioCutter
from config import Config


def main():
    audio_path = '' #FIXME
    subsegments_path = AudioCutter(audio_path).cut()
        
    Helpers.move_files(subsegments_path, Config.dataset_root)

if __name__ == "__main__":
    main()