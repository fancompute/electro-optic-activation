# Electro-optic Activation Function

## Overview

This repository contains the notebooks for training optical neural networks (ONNs) with the electro-optic activation function proposed in:

* I. A. D. Williamson, T. W. Hughes, M. Minkov, B. Bartlett, S. Pai, and S. Fan, "[Reprogrammable Electro-Optic Nonlinear Activation Functions for Optical Neural Networks](https://doi.org/10.1109/JSTQE.2019.2930455)," IEEE Journal of Selected Topics in Quantum Electronics, Jul. 2019.

The code for generating the main results of the above paper are provided in the Jupyter notebooks in this repository. See the section below for more details on each of the notebooks. **Please consider citing the above paper if you use any of the code in this repository.**

## Contents

* `study_characterization.ipynb` - Activation function characterization results (some of the initial figures in the paper).
* `study_xor.ipynb` - Training results for the XOR logic function task (results in Fig. 5 of the paper). **Note:** this uses the [neuroptica](https://github.com/fancompute/neuroptica) ONN simulator framework.
* `study_mnist.ipynb` - Training results for the MNIST image recognition task (results in Fig. 6 of the paper). **Note:** this uses the [neurophox](https://github.com/solgaardlab/neurophox/) ONN simulator framework.

## Note on result consistency

You will need to set the `PYTHONHASHSEED` environment variable (in addition to setting the seeds for the various packages *inside* the notebook). For example, you can run jupyter lab like so:

`PYTHONHASHSEED=0 jupyter lab`

**Note:** for some reason this still doesn't give *completely* consistent results from run to run.
