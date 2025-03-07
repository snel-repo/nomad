import logging, json
from os import path
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from align_tf2.validation import utils
from align_tf2 import compat

def plot_fitlogs(lfl_search_dir,
                 lf2_search_dir,
                 plot_field,
                 smooth=False,
                 logy=False,
                 save_dir=None):
    """Plots the training logs from LFL and LF2 alignment random 
    searches. Note that the runs are not paried so be sure to check
    that the searches are comparable before drawing conclusions.

    Parameters
    ----------
    lfl_search_dir : str
        The path to the LFL alignment random search.
    lf2_search_dir : str
        The path to the LF2 alignment random search.
    plot_field : str
        The field to plot.
    smooth : bool, optional
        Whether to smooth data before plotting, by default False
    logy : bool, optional
        Whether to log-transform data before plotting, by default False
    save_dir : str, optional
        Where to save the plot, by default None
    """
    # load the data and arrange it in the appropriate format
    lfl_fitlogs = compat.read_random_search_fitlogs(lfl_search_dir)
    lf2_fitlogs = utils.read_random_search_fitlogs(lf2_search_dir)
    # LF2 only records at the end of each epoch
    x_field = 'epoch'
    lfl_plot_df = lfl_fitlogs.pivot(x_field, 'trial_id', plot_field)
    lf2_plot_df = lf2_fitlogs.pivot(x_field, 'trial_id', plot_field)
    # perform calculations if needed
    field_label = plot_field
    if smooth:
        lfl_plot_df = lfl_plot_df.rolling(
            20, win_type='gaussian').mean(std=20)
        lf2_plot_df = lf2_plot_df.rolling(
            20, win_type='gaussian').mean(std=20)
        field_label = 'smth_' + field_label
    if logy:
        lfl_plot_df = np.log10(lfl_plot_df)
        lf2_plot_df = np.log10(lf2_plot_df)
        field_label = 'log_' + field_label
    # plot and add labels
    fig, ax = plt.subplots(figsize=(11,6))
    alpha = 0.4
    lfl_plot_df.plot(label='LFL', color='b', alpha=alpha, legend=False, ax=ax)
    lf2_plot_df.plot(label='LF2', color='r', alpha=alpha, legend=False, ax=ax)
    plt.xlabel(x_field)
    plt.ylabel(field_label)
    plt.title(
        "Train log for alignment validation\n"
        f"LFL: {lfl_search_dir}\n"
        f"LF2: {lf2_search_dir}")
    # show or save the figure
    if save_dir == None:
        plt.show()
    else:
        filename=plot_field.replace('.', '_').lower()
        fig_path = path.join(save_dir, f'{filename}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    LFL_SEARCH_DIR = (
        '/snel/home/lwimala/tmp/ray_results/'
        'nomad_randomSearch_200317_LRI_LDF_100bin_overlap_fixedperc_BS_2000'
    )
    LF2_SEARCH_DIR = (
        '/snel/home/yali/ray_results/'
        'nomad_200703_random_search_LRI_MGN_jitter4'
    )
    # Plot the validation KL from these runs as a demonstration
    plot_fitlogs(LFL_SEARCH_DIR, LF2_SEARCH_DIR, 'val_kl', logy=True)
