# The paper will be published :)
# STIPoser: Spatial Temporal Inertia Poser with 6 IMU sensors

Code for my bachelor project at [FUM CARE](https://fum-care.com/)
## Acknowledgement
Some of our codes are adapted from [PIP](https://github.com/Xinyu-Yi/PIP) and [Dynaip](https://github.com/dx118/dynaip.git).

## Install requirements
I tested my model on Windows OS with python 3.9
```shell
pip install -r requirements.txt
```

## Dataset 
We used public IMU and Motion Capture dataset to train our model.
1. Download DIP-IMU and Total Capture datasets from [here](https://dip.is.tue.mpg.de/).
2. Download Amass datasets from [here](https://amass.is.tue.mpg.de/).
3. Download Emonike dataset from [here](https://zenodo.org/records/7821844).
4. Download AnDy (xsens_mvnx.zip) dataset from [here](https://zenodo.org/records/3254403).
5. Download CIP (MTwAwinda.zip) dataset from [here](https://doi.org/10.5281/zenodo.5801928).

Download the above datasets, extract and place them in `./data/raw data/`.
```python
data
├─raw data
│  ├─AMASS
│  │  ├─ACCAD
│  │  ├─...
│  ├─DIP_IMU
│  │  ├─s_...
│  │  ├─...
│  ├─Emonike
│  │  ├─Data
│  │  ├─...
│  ├─MTwAwinda
│  │  ├─trials
│  │  ├─...
│  ├─TotalCapture
│  │  ├─TotalCapture (AMASS dataset)
│  │  ├─TotalCapture_Real_60_FPS (DIP dataset)
│  └─xsens_mvnx
│  │  ├─Participant_...
│  │  ├─...
└─smpl
```
Then set path for each dataset in preprocess file.
```python
amass_path = "./data/raw data/AMASS"
dip_path = "./data/raw data/DIP_IMU"
tc_path = "./data/raw data/TotalCapture"
emonike_path = "./data/raw data/EmokineDataset_v1.0/Data/MVNX"
mtw_path = "./data/raw data/MTwAwinda"
mvnx_path = "./data/raw data/xens_mnvx/"
```

## SMPL Model
Download SMPL model from [here](https://smpl.is.tue.mpg.de/). 
You should click `SMPL for Python` and download the `version 1.0.0 for Python 2.7 (10 shape PCs)`. 
Then unzip and place them in `./data/smpl/`.
```python
data
├─raw data
└─smpl
│  ├─SMPL_MALE.pkl
│  ├─SMPL_FEMALE.pkl
```
Then set your smpl path in files. 
```python
smpl_path = "./data/smpl/SMPL_MALE.pkl"
```


## Preprocess and Extract Sequence
You should preprocess the datasets before training and validation.
Then extract training sequences for training.

```shell
python preprocess.py
python extract_data_seq.py
```

## Configs
Set your training configs in `configs.py`.

## Train 
Run `main.py` to train model.
```shell
python main.py
```

##  Visualization
You can visualize one sample of a dataset with `visualize.py`.
Just give your path base on this code

```python
files = glob.glob(r"./data/Valid/TC*/*/*.npz")
...
model = resume(model=model, path="""best model path""")
visualize(model, data_file=files["""idx of your data in dataset"""])
```


##  Validation
You can validate a model with `evaluator.py`.
Just give your path base on this code

```python
files = glob.glob(r"./data/Valid/TC*/*/*.npz")
...
model = resume(model=model, path="""best model path""")

evaluator = PoseEvaluator(model=model, data_files=files, configs=model_config)
evaluator.run()
```
