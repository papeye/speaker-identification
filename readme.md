# Speaker identification ![tests](https://github.com/papeye/speaker-identification/actions/workflows/python-package.yml/badge.svg)

Implementation of CNN for speaker identification in TensorFlow. Trained on the example data of 20 speakers, the network correctly identifies about 90% of the speakers. 

## Setup

To install the package run
```
pip install git+https://github.com/papeye/speaker-identification.git@setup-package
```

The example demonstrating the preformance of the package is found in example.py. 


## Usage

The package provides the class SpeakerIdentifier, which initializes a CNN for speaker identification. This class requires a path to a directory containing training audio files in .wav format. The audio file names should represent the speakers' names:
```
user1 = SpeakerIdentifier(model_name=<model_name>, training_ds_dir=<train_data_dir>)
```
>[!NOTE]
>Only single file per speaker is supported for training at the moment

>[!IMPORTANT]
>It is assumed that only one person is speaking in an audio file

 
The audio files are subsequently processed by optional Voice Activity Detection (optional) to remove silence and split into 1s segments which are used for the training. Optional data augmentation is possible by adding noise to the training audios (configurable in Config file).

The training is done via train method of SpeakerIdentifier:
```
user1.train(train_data_dir=<train_example_dir>, with_vad=True)
```

The trained model can be used to predict the speaker from the audio file by calling

```
predictions = user1.predict(test_data_dir=<test_example_dir>, with_vad=True)
```

which returns a class Result mapping each speaker directory to a dictionary of predicted speaker labels and their normalized probabilities, sorted in descending order of probability. Class Results has best_prediction getter which returns the most probable speaker.

## For contributors

The number of epochs can be adjusted in the Config file, but on the example audio 90% accuracy is obtained for only 3 epochs (which is default). Convolutional neural network is defined in nnmodel.py file using residual blocks and its size is chosen for identifying 20 speakers but users are encouraged to experiment with hyperparameters for the best performance. If available, weights from previous training run will be used as for initialization of the NN with replaced output layer.

If user provides less then 20 speakers, base speakers are added to the training dataset to achive the best consistent performance. However if user provides more than 20 speakers in the training_ds_dir no base speakers will be added to the training dataset. 
