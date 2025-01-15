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

## Functionality

The package provides the class SpeakerIdentifier, which initializes a CNN for speaker identification. This class requires a path to a directory containing training audio files. The audio file names should represent the speakers' names:
```
user1 = SpeakerIdentifier(model_name="user1",training_ds_dir=train_example_dir)
```

 Note that in the current version, only single files are used for training and it is assumed that in each of the audio files, only one speaker is present. The audio files are subsequently processed by VAD (optional) to remove unnecessary parts of audio and split into 1s segments which are used for the training. Optional data augmentation is possible by adding noise to the training audios (set in Config file).

The training is done via train method of SpeakerIdentifier:
```
user1.train(train_data_dir=train_example_dir, with_vad=True)
```
with the number of epochs set in the Config file, default is 3. Convolutional neural network is defined in nnmodel.py file using residual blocks and its' size is chosen for identifying 20 speakers but users are encouraged to experiment with hyperparameters for the best performance. If user provides less then 20 speakers, base speakers are added to the training dataset to achive the best consistent performance. However if user provides more than 20 speakers in the training_ds_dir no base speakers will be added to the training dataset. The trained model can be used to predict the speaker from the audio file by calling

```
predictions = user1.predict(test_data_dir=test_example_dir, with_vad=True)
```

which returns a class mapping each speaker directory to a dictionary of predicted speaker labels and their normalized probabilities, sorted in descending order of probability. 

