import shutil
import os

class Helpers: 
    @staticmethod
    def move_files(source, target):
        '''Copies all files from source to target directory'''
        
        if not os.path.exists(target):
          os.makedirs(target)
          
        for filename in os.listdir(source):
            source_path = os.path.join(source, filename)
            destination_path = os.path.join(target, filename)
            shutil.copy(source_path, destination_path)