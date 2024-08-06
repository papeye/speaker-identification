from data_preprocessing.audio_cutter import AudioCutter


def main():
    audio_path = 'example_data/ryczekWav.wav'
    AudioCutter(audio_path).cutAndAddToBaseData()
    

if __name__ == "__main__":
    main()