from data_preprocessing.audio_cutter import AudioCutter
import os

os.environ["KERAS_BACKEND"] = "tensorflow"


class SpeakerIdentifier:
    
    def learn(self, audio_path):
        #TODO implement learning
        pass
    
    def identify(self, audio_path):
        #TODO implement identification
        pass
        