# Nonlinear Manifold Alignment with Dynamics (NoMAD)
If you use NoMAD in published work, please cite our manuscript: 
> Karpowicz BM, Ali YH, Wimalasena LN, Sedler AR, Keshtkaran MR, Bodkin K, Ma X, Rubin DB, Williams ZM, Cash SS, Hochberg LR, Miller LE, Pandarinath C. Stabilizing brain-computer interfaces through alignment of latent dynamics. bioRxiv preprint. doi:10.1101/2022.04.06.487388. 2022 Nov 08.

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
Example usage of the NoMAD codebase along with sample data files are described in `demo/demo.ipynb`.
