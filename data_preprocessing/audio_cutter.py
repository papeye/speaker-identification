from pyannote.audio import Pipeline
import onnxruntime
from pydub import AudioSegment
import os

from data_preprocessing.models.segment import Segment
from config import Config


class AudioCutter:
    '''
    Takes passed audio_path and:
        1. Diarizes an audio (returns dict with times when speaker speaks)
        2. Cuts audio into segments based on diarization
        3. Cuts segments into subsegment_length subsegments
        4. Saves subsegments to output_path
    ''' 
    
    def __init__(self, audio_path, subsegment_length = 1000):
        self.audio_path = audio_path
        self.output_path = Config.subsegments_path
        self.subsegment_length = subsegment_length


    
    def __diarize(self):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0", use_auth_token=Config.hugging_face_token
        )

        #send pipeline to GPU (when available)
        # import torch
        # pipeline.to(torch.device("cuda"))

        diarization = pipeline(self.audio_path)


        segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
          start = int(turn.start * 1000)
          end = int(turn.end * 1000)

          if end - start >= self.subsegment_length:
            segments.append(Segment(start, end, self.audio_name))

        
        print('Diarization:')
        print(segments[:5])
        print(f'...and so on till {len(diarization)}')
        
        return segments
        
    
    def __segment(self, segments):
        subsegments = []
        
        for segment in segments:
          start = segment.start
          end = segment.end

          s_start = start
          s_end = start + self.subsegment_length

          while(s_end < end):
            subsegments.append(Segment(s_start, s_end, segment.speaker))
            s_start = s_start + self.subsegment_length
            s_end = s_start + self.subsegment_length

        print('Segmentation:')
        print(subsegments[:5])
        print(f'and so on till {len(subsegments)}')
        
        return subsegments
    
    
    def __save_subsegments(self, subsegments):
        audio = AudioSegment.from_wav(self.audio_path)
        output = os.path.join(self.output_dir, 'subsegments')

        if not os.path.exists(output):
          os.makedirs(output)


        for segment in subsegments:
          start = segment.start
          end = segment.end
          file_name = f"{start} - {end}.wav"

          segment_output = os.path.join(output, file_name)
          new_audio = audio[start:end]
          new_audio.export(segment_output, format="wav")
          print(f"Saved {file_name}")
        
        
        
    def cut(self):
        diarization = self.__diarize()
        subsegments = self.__segment(diarization)
        self.__save_subsegments(subsegments)
        return self.output_path