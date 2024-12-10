import os, shutil, time
from pydub import AudioSegment
from typing import List
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from .models.segment import Segment
from ..config import Config


def cut_all_into_segments(
    audios_dir: str,
    output_dir: str,
    subsegment_length: int = 1000,
    with_vad: bool = True,
) -> None:

    for audio in os.listdir(audios_dir):
        audio_path = os.path.join(audios_dir, audio)
        AudioCutter(audio_path, output_dir, subsegment_length).cut(with_vad)

    print(
        f"All audios from {audios_dir} cut into segments of length {subsegment_length/ 1000}s and saved to ",
        output_dir,
    )


class AudioCutter:
    """
    Takes passed audio_path and:
        1. Diarizes an audio (returns dict with times when speaker speaks)
        2. Cuts audio into segments based on diarization
        3. Cuts segments into subsegment_length subsegments
        4. Saves subsegments to output_path
    """

    def __init__(
        self,
        audio_path: str,
        output_path: str,
        subsegment_length: int = 1000,
    ) -> None:
        self.audio_path = audio_path
        self.audio_name = os.path.basename(audio_path)
        self.audio = AudioSegment.from_wav(audio_path)
        self.output_path = os.path.join(output_path, self.audio_name)
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)
        self.subsegment_length = subsegment_length

    def __simple_cut(self) -> List[Segment]:
        """Cut audio into segments of subsegment_length length"""
        start = 0.0
        audio_duration = len(self.audio)

        print(f"Audio cut into 1 segment of length {audio_duration / 1000}s")

        return [Segment(start, audio_duration, self.audio_name)]

    def __detect_voice_activity(self) -> List[Segment]:
        """Detect voice activity in audio using Silero VAD model"""
        start_time = time.time()

        # Load Silero VAD model
        model = load_silero_vad()

        # Read audio file
        wav = read_audio(self.audio_path)

        # Get speech timestamps from the audio
        speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)

        # Create segments from the speech timestamps
        segments = [
            Segment(int(ts["start"] * 1000), int(ts["end"] * 1000), self.audio_name)
            for ts in speech_timestamps
        ]

        print(
            f"Voice activity detection took {time.time() - start_time} seconds and found {len(segments)} segments"
        )

        return segments

    def __segment(self, segments: List[Segment]) -> List[Segment]:
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
            f"Audio {self.audio_name} cut into {len(subsegments)} subsegments of length {self.subsegment_length / 1000}s"
        )

        return subsegments

    def __save_subsegments(self, subsegments: List[Segment]) -> None:

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        for segment in subsegments:
            start = segment.start
            end = segment.end
            file_name = f"{start} - {end}.wav"

            segment_output = os.path.join(self.output_path, file_name)
            new_audio = self.audio[start:end]
            new_audio = new_audio.set_frame_rate(Config.sampling_rate)
            new_audio = new_audio.set_sample_width(Config.sample_width)
            new_audio.export(segment_output, format="wav")

        print(f"Saved {len(subsegments)} subsegments to {self.output_path}")

    def cut(self, with_vad: bool = True) -> None:
        segments = self.__detect_voice_activity() if with_vad else self.__simple_cut()

        subsegments = self.__segment(segments)
        self.__save_subsegments(subsegments)
