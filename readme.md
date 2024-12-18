Implementation of CNN for speaker identification in TensorFlow. Trained on the example data of 52 speakers, the network correctly identifies about 85% of the speakers. 

# Running instructions

To install the package run
```
pip install git+https://github.com/papeye/speaker-identification.git@setup-package
```
and install the dependencies with
```
pip install -r requirements.txt
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

### rebase instructions
go to master and download latest version:
```
git checkout master
git pull origin master
```
merge the branch to the master
```
git merge origin/master
```
go through all the commits which give conficts. If there are problems with files such as model.keras then:
```
git add .
```
and continue with 
```
git rebase --continue
```
until rebase is complete. You can then force push it:
```
git push -f branch-name
```
