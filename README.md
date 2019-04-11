# PB-Net
This repo is a minimum example of PeakBoundaryNet: a sequential neural network for peak boundary detection and peak area integration. 

The data used for test is stored in `test_input.pkl` as a list of peak instances, each instance is a tuple of (X, y, metainfo, sample name & peak name).

Predictions by sequential and reference-based PB-Nets are respectively stored in `test_preds.pkl` and `test_preds_ref.pkl`, as a list of numpy arrays representing peak start/end probabilities (in the same order as input). These predictions can also be generated through `run.py`, based on the pre-trained models.

`plotting.py` will generate the figures and tables used in the manuscript. `human_annotators.py` will generate the comparison figure/numbers between PB-Nets and group of annotators(Figure 4). `skyline_corrs.py` will generate the abundance metrics for skyline predictions.

## Requirements
* pandas
* sklearn
* numpy
* scipy
* matplotlib
* pytorch
