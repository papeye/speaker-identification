import os
import shutil
from config import Config
from pathlib import Path



class DataPreparator:
    '''
    The dataset is composed of 7 folders, divided into 2 groups:

        - Speech samples, with 5 folders for 5 different speakers. Each folder contains
        1500 audio files, each 1 second long and sampled at 16000 Hz.
        - Background noise samples, with 2 folders and a total of 6 files. These files
        are longer than 1 second (and originally not sampled at 16000 Hz, but we will resample them to 16000 Hz).
        We will use those 6 files to create 354 1-second-long noise samples to be used for training.

        Let's sort these 2 categories into 2 folders:

        - An `audio` folder which will contain all the per-speaker speech sample folders
        - A `noise` folder which will contain all the noise samples 
        
        
        Before sorting the audio and noise categories into 2 folders, we have the following directory structure:
        main_directory/
        ...speaker_a/
        ...speaker_b/
        ...speaker_c/
        ...speaker_d/
        ...speaker_e/
        ...other/
        ..._background_noise_/
        
        
        After sorting, we end up with the following structure:
        main_directory/
        ...audio/
        ......speaker_a/
        ......speaker_b/
        ......speaker_c/
        ......speaker_d/
        ......speaker_e/
        ...noise/
        ......other/
        ......_background_noise_/
          
    '''
    

    def __init__(self):
        pass
        
          
    def ___move_files_to_proper_folders(self):
        for folder in os.listdir(Config.dataset_root):
            if os.path.isdir(os.path.join(Config.dataset_root, folder)):
                if folder in [Config.audio_subfolder, Config.noise_subfolder]:
                    # If folder is `audio` or `noise`, do nothing
                    continue
                elif folder in ["other", "_background_noise_"]:
                    # If folder is one of the folders that contains noise samples,
                    # move it to the `noise` folder
                    shutil.move(
                        os.path.join(Config.dataset_root, folder),
                        os.path.join(Config.dataset_noise_path, folder),
                    )
                else:
                    # Otherwise, it should be a speaker folder, then move it to
                    # `audio` folder
                    shutil.move(
                        os.path.join(Config.dataset_root, folder),
                        os.path.join(Config.audio_path, folder),
                    )
            
    def __resample(self, folder_path):
        command = (
        "for dir in `ls -1 " + folder_path + "`; do "
        "for file in `ls -1 " + folder_path + "/$dir/*.wav`; do "
        "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
        "$file | grep sample_rate | cut -f2 -d=`; "
        "if [ $sample_rate -ne 16000 ]; then "
        "ffmpeg -hide_banner -loglevel panic -y "
        "-i $file -ar 16000 temp.wav; "
        "mv temp.wav $file; "
        "fi; done; done"
        )
        os.system(command)
                
         
    def __prepare_noise(self):
        '''
        We load all noise samples (which should have been resampled to 16000)
        We split those noise samples to chunks of 16000 samples which correspond to 1 second duration each
        '''
        
        # Get the list of all noise files
        noise_paths = []
        for subdir in os.listdir(Config.dataset_noise_path):
            subdir_path = Path(Config.dataset_noise_path) / subdir
            if os.path.isdir(subdir_path):
                noise_paths += [
                    os.path.join(subdir_path, filepath)
                    for filepath in os.listdir(subdir_path)
                    if filepath.endswith(".wav")
                ]
        if not noise_paths:
            raise RuntimeError(f"Could not find any files at {Config.dataset_noise_path}")
        print(
            "Found {} files belonging to {} directories".format(
                len(noise_paths), len(os.listdir(Config.dataset_noise_path))
            )
        )
             
        self.__resample(Config.dataset_noise_path)
        
        
        
    def __prepare_new_speaker(self):
      self.__resample(Config.)
        