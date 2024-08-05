import shutil
import os

class Helpers: 
    @staticmethod
    def move_files(new_cuts_folder, base_dataset):
        for filename in os.listdir(new_cuts_folder):
            source_path = os.path.join(new_cuts_folder, filename)
            destination_path = os.path.join(base_dataset, filename)
            shutil.copy(source_path, destination_path)