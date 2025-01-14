# Speaker identification

Implementation of CNN for speaker identification in TensorFlow. Trained on the example data of 20 speakers, the network correctly identifies about 90% of the speakers. 

## Running instructions

To install the package run
```
pip install git+https://github.com/papeye/speaker-identification.git@setup-package
```

### Create venv
```
python -m venv venv
```

### Activate venv
```
.\venv\Scripts\Activate.ps1
```

### install requirements
```
pip install -r requirements.txt
```
### Example run on 20 speakers using 
```
python .\example.py
```

## Funcionality 

The package provides the class SpeakerIdentifier, which initializes a CNN. This class takes a path to the directory with training audio files, with the files' names denoting the speakers' names. Note that in the current version, only single files are used for training and it is assumed that in each of the audio files, only one speaker is present. The audio files are subsequently processed by VAD (optional) to remove unnecessary parts of audio and split into 1s segments which are used for the training.

The training is done via train method of SpeakerIdentifier, with the number of epochs set in the Config file. 

