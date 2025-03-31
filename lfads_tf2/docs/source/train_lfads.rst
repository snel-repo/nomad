##########
Training Example
##########

The `train_lfads.py` script is both a demo for those who are 
new to running LFADS and a test script to rapidly test changes. 
The main steps to running LFADS are:

    1. Create a configuration YAML file to overwrite any or
        all of the defaults
    2. Create an LFADS object by passing the path to the 
        configuration file.
    3. Train the model using `model.train`
    4. Perform posterior sampling to create the posterior 
        sampling file using `model.sample_and_average`
    5. Load rates, etc. for further processing using 
        `lfads_tf2.utils.load_posterior_averages`
