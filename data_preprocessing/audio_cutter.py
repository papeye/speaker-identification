from typing import List
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import shutil
import time

from data_preprocessing.models.segment import Segment
from config import Config


class AudioCutter:
    """
    Takes passed audio_path and:
        1. Diarizes an audio (returns dict with times when speaker speaks)
        2. Cuts audio into segments based on diarization
        3. Cuts segments into subsegment_length subsegments
        4. Saves subsegments to output_path
    """

    def __init__(self, audio_path, subsegment_length=1000):
        self.audio_path = audio_path
        self.audio_name = os.path.basename(audio_path)
        self.output_path = os.path.join(Config.dataset_train_audio, self.audio_name)
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)
        self.subsegment_length = subsegment_length

    def __diarize(self):
        start_time = time.time()
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0", use_auth_token=Config.hugging_face_token
        )

        # send pipeline to GPU (when available)
        # import torch
        # pipeline.to(torch.device("cuda"))

        diarization = pipeline(self.audio_path)

        segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = int(turn.start * 1000)
            end = int(turn.end * 1000)

            if end - start >= self.subsegment_length:
                segments.append(Segment(start, end, self.audio_name))

        print(f"Diarization took {time.time() - start_time} seconds and forund {len(segments)} segments")

        return segments

    def __segment(self, segments: List[Segment]):
        subsegments = []

        for segment in segments:
            start = segment.start
            end = segment.end

            s_start = start
            s_end = start + self.subsegment_length

            while s_end < end:
                subsegments.append(Segment(s_start, s_end, segment.speaker))
                s_start = s_start + self.subsegment_length
                s_end = s_start + self.subsegment_length

        print(f"Audio cut into {len(subsegments)} subsegments of length {self.subsegment_length * 1000}s")

        return subsegments

    def __save_subsegments(self, subsegments: List[Segment]):
        audio = AudioSegment.from_wav(self.audio_path)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        for segment in subsegments:
            start = segment.start
            end = segment.end
            file_name = f"{start} - {end}.wav"

            segment_output = os.path.join(self.output_path, file_name)
            new_audio = audio[start:end]
            new_audio.export(segment_output, format="wav")
        
        print(f"Saved {len(subsegments)} subsegments to {self.output_path}")

    def cutAndAddToBaseData(self):
        diarization = self.__diarize()
        subsegments = self.__segment(diarization)
        self.__save_subsegments(subsegments)
