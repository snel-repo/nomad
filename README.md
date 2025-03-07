# nomad
Code for Nonlinear Manifold Alignment with Dynamics (NoMAD) detailed in Karpowicz et al., Nat Comms in press

conda create -n nomad python=3.7.7
conda activate nomad 
conda install -c nvidia cudatoolkit=10.0 
conda install -c nvidia cudnn=7.6
cd lfads_tf2
pip install -e .
cd ..
cd nomad
pip install -e .
