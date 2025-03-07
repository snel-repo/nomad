from os import path
from glob import glob
import pandas as pd
import yaml

from lfads_tf2.utils import flatten

def read_random_search_hps(search_dir):
    """Reads the hyperparameters of all models in an align_tf2 alignment
     random search.

    Parameters
    ----------
    search_dir : str
        The path to the random search.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all hyperparameters used in the search.
    """

    # Create the pattern to match the model directories
    run_dir_pattern = path.join(search_dir, 'tuneAlign*/align_dir')
    # Find all of the matching run directories
    run_dirs = sorted(glob(run_dir_pattern))
    hps = []
    for i, run_dir in enumerate(run_dirs):
        hps_file = path.join(run_dir, 'align_spec.yaml')
        hps_dict = flatten(yaml.full_load(open(hps_file)))
        hps_dict['trial_id'] = i
        hps.append(hps_dict)
    hps = pd.DataFrame(hps)

    return hps

def read_random_search_fitlogs(search_dir):
    """Reads the fitlogs of all models in an align_tf2 alignment 
    random search.

    Parameters
    ----------
    search_dir : str
        The path to the random search.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the fitlogs of all models in the search
    """

    # Create the pattern to match the model directories
    run_dir_pattern = path.join(search_dir, 'tuneAlign*/align_dir')
    # Find all of the matching run directories
    run_dirs = sorted(glob(run_dir_pattern))
    fitlogs = []
    for i, run_dir in enumerate(run_dirs):
        # Read and number the data in each fitlog
        train_data_file = path.join(run_dir, 'train_data.csv')
        fitlog = pd.read_csv(train_data_file)
        fitlog['trial_id'] = i
        fitlogs.append(fitlog)
    fitlogs = pd.concat(fitlogs)

    return fitlogs