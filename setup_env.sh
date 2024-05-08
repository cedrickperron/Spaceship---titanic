#!/bin/bash

# Load Python 3.9.10 module
module load python

# Create a virtual environment with system site packages enabled
virtualenv --system-site-packages ./.virtualenvs/jupteach

# Activate the virtual environment
source ./.virtualenvs/jupteach/bin/activate

# Install Jupyter and JupyterLab
#pip install jupyter jupyterlab
pip install --upgrade pip


# Install required packages
pip install jupyter jupyterlab numpy h5py joblib pennylane seaborn pandas scikit-learn matplotlib tensorflow

# Deactivate the virtual environment
deactivate


## >>> bash setup_env.sh
