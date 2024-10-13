from typing import List
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import shutil
import time
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection


from data_preprocessing.models.segment import Segment
from config import Config


def cut_all_into_segments(audios_dir, output_dir, subsegment_length=1000):

    for audio in os.listdir(audios_dir):
        audio_path = os.path.join(audios_dir, audio)
        _AudioCutter(audio_path, output_dir, subsegment_length).cutAndAddToBaseData()
        print(
            f"All audios from {audios_dir} cut into segments of length {subsegment_length/ 1000}s and saved to ",
            output_dir,
        )


class _AudioCutter:
    """
    Takes passed audio_path and:
        1. Diarizes an audio (returns dict with times when speaker speaks)
        2. Cuts audio into segments based on diarization
        3. Cuts segments into subsegment_length subsegments
        4. Saves subsegments to output_path
    """

    def __init__(self, audio_path, output_path, subsegment_length=1000):
        self.audio_path = audio_path
        self.audio_name = os.path.basename(audio_path)
        self.output_path = os.path.join(output_path, self.audio_name)
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)
        self.subsegment_length = subsegment_length

    def __diarize(self):
        start_time = time.time()

        model = Model.from_pretrained(
            "pyannote/segmentation", use_auth_token=Config.hugging_face_token
        )

        pipeline = VoiceActivityDetection(segmentation=model)

        HYPER_PARAMETERS = {
            # onset/offset activation thresholds
            "onset": 0.5,
            "offset": 0.5,
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 1.0,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.0,
        }
        pipeline.instantiate(HYPER_PARAMETERS)
        diarization = pipeline(self.audio_path)

        segments = [
            Segment(int(speech.start * 1000), int(speech.end * 1000), self.audio_name)
            for speech in diarization.get_timeline().support()
        ]

        print(
            f"Diarization took {time.time() - start_time} seconds and found {len(segments)} segments"
        )

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

        print(
            f"Audio cut into {len(subsegments)} subsegments of length {self.subsegment_length / 1000}s"
        )

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
            new_audio = new_audio.set_frame_rate(Config.sampling_rate)
            new_audio.export(segment_output, format="wav")

        print(f"Saved {len(subsegments)} subsegments to {self.output_path}")

    def cutAndAddToBaseData(self):
        diarization = self.__diarize()
        subsegments = self.__segment(diarization)
        self.__save_subsegments(subsegments)
