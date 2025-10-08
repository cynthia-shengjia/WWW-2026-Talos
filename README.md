# IR-Benchmark

This repository contains the code for the submission to the WWW 2026.


# Conda Environment

We provide the environment requirements, you can construct the environment as:
```
conda env create -f environment.yaml
```

This will create a conda environment named `rec_gym`. To activate the environment, run:
```
conda activate rec_gym
```

## Code Structure

The code is organized as follows:

```
IR-Benchmark-SLatK
│   README.md                           # This file
|   main.py                             # The model training script for SL, BSL, LLPAUC, etc.
|   mainAdvInfoNCE.py                   # The model training script for AdvInfoNCE.
|   mainTopKRegression.py               # The model training script for Talos, and so on.
|   Config                              # The nni auto hyperparameter config
|   |   Talos.yaml                      # The hyperparameter tuning framework for Talos
│   normal_data                         # Datasets
│   │   IID_Data_Used                   # IID dataset
|   |   |   Beauty                      # Beauty IID dataset
|   |   ... (other datasets)            # Other IID datasets
│   optimizer                           # Talos and other baseline methods
|   |   optim_QR.py                     # Talos Loss
|   |   ... (other baseline methods)    # Other modules
|   dataset                             # The dataloader
|   model                               # The recommendation backbone
|   tools                               # Training tools, e.g., calculating metrics
```


## NNI Hyperparameter Tuning

We provide automatically conduct hyperparameter tuning with NNI framework. You can run:
```
nnictl create --config ./Config/Talos.yaml --port [Your Possible Port]
```