# LocHeatmap: Differentiable Heatmap Regression for Signal Source Localization with Radio Map Completion

## Description

**LocHeatmap** is a sophisticated deep learning framework designed to accurately localize non-cooperative signal sources using sparse RSS (Received Signal Strength) and AoA (Angle of Arrival) data collected by UAVs (Unmanned Aerial Vehicles). This method leverages a Modified Deep Completion Autoencoder (Modified-DCAE) to complete the sparse RSS and AoA data into comprehensive radio maps. Subsequently, the LocUNet model transforms these radio maps into a heatmap, which is then subjected to differentiable heatmap regression to precisely estimate the location of the signal source.

## Training or Running

For training:

1. download the dataset at [?][?]
2. install requirement.txt using `pip install requirement.txt`
3. running `aoa_model.py rss_model.py heatmap_transform.py` in order
4. (optional) if you want to do the evaluation, you need to train all the other baseline and ablation experiment(you need to copy the whole `*.py` into the main directory) 
5. running main.py (you need to comment the untraining model first in the `utils_model.py`)

For running:

1. download the dataset at [?][?]
2. download the models at [?][?]
3. install requirement.txt using `pip install requirement.txt`
4. modify and running `evaluation.py`

The mode is listed as followed

| MODE NAME                    | DESCRIPTION                                   |
| ---------------------------- | --------------------------------------------- |
| proposal                     | proposal method                               |
| DCAe                         | baseline1: DCAe                               |
| DCAe_simple                  | baseline2: Modified-DCAe                      |
| LocUNet                      | baseline3: LocUNet                            |
| LocUNet_softargmax           | baseline4: Modified-LocUNet                   |
| proposal_rss_only            | proposal method (only use rss)                |
| proposal_aoa_only            | proposal method (only use aoa)                |
| proposal_gaussian_dsnt       | proposal method (using gaussian + dsnt)       |
| proposal_gaussian_softargmax | proposal method (using gaussian + SoftArgmax) |
| proposal_dsnt                | proposal method (using dsnt)                  |

