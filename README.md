# Nonlinear Manifold Alignment with Dynamics (NoMAD)
If you use our 
Code for Nonlinear Manifold Alignment with Dynamics (NoMAD) detailed in Karpowicz et al., Nat Comms in press

## Installation
To create an environment and install the dependencies of the project, run the following commands:

```
conda create -n nomad python=3.7.7
conda activate nomad 
conda install -c nvidia cudatoolkit=10.0 
conda install -c nvidia cudnn=7.6
cd lfads_tf2
pip install -e .
cd ..
cd nomad
pip install -e .
```

## Usage
Example usage of the NoMAD codebase along with sample data files are described in `demos/demos.ipynb`.