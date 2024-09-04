import os
class Config:
    hugging_face_token = "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"
    dataset_root = '16000_pcm_speeches'
    sampling_rate=16000
    audio_subfolder = "audio"
    noise_subfolder = "noise"
    
    dataset_audio_path = os.path.join(dataset_root, audio_subfolder)
    dataset_noise_path = os.path.join(dataset_root, noise_subfolder)
    
