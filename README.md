# DSB2017 PyTorch
Tested on Python 3.5+, pytorch version 0.4.1 in Windows 10, with or without GPU.

## Prerequisites:
* SimpleITK
* dicom
* scipy
* pytorch
* scikit-image
* nvidia-ml-py
* matplotlib
* h5py

### Ubuntu /Linux prerequisites
`sudo apt-get install python3-tk`



# Instructions for running

First unzip the stage 1/2 data.
### Step 1

Preprocess DSB stage1/2 or alike data for inferencing.
(doesn't need annotation).


```
python inference_1_preprocess.py --data dir/to/stage1or2 --save dir/to/save/npyfiles
```
### Step 2
Predict nodules locations, generate _pbb.npy and _lbb.npy files for each patient.


```
python inference_2_n_net.py --data dir/to/stage1or2 --save dir/to/save/npyfiles
```
### Step 3
Generate cancer prediction and save to a CSV file.


```
python inference_3_c_net.py --data dir/to/stage1or2 --save dir/to/save/npyfiles --csv path/to/predicts.csv
```

### Step 4 Visualization

Require [Python 3.5+](https://www.python.org/ftp/python/3.6.4/python-3.6.4.exe) and [Jupyter notebook](https://jupyter.readthedocs.io/en/latest/install.html) installed.

In the project start a command line run
```
jupyter notebook
```
In the opened browser window open
```
detection result demo.ipynb
```