from data_preprocessing.audio_cutter import AudioCutter
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import shutil

class SpeakerIdentifier:
    
    def __move_files(new_cuts_folder, base_dataset):
        for filename in os.listdir(new_cuts_folder):
            source_path = os.path.join(new_cuts_folder, filename)
            destination_path = os.path.join(base_dataset, filename)
            shutil.copy(source_path, destination_path)
    
    
    def learn(self, audio_path):
        subsegments_path = AudioCutter(audio_path).cut()
        
        self.__move_files(subsegments_path, '16000_pcm_speeches')
        