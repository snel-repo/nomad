import h5py    
import pdb
import numpy as np    
from os import path
import tensorflow as tf 
import matplotlib.pyplot as plt 
from helper_funcs import shuffle_spikes_in_time
tf.set_random_seed(42)
np.random.seed(42)

# make a fake dataset 
train_truth = np.zeros((50, 29))

# assign one time point to have a high spike count 
train_truth[0, :] = 8.0
train_truth = np.expand_dims(train_truth, 0)

# run spike jitter with different widths 
widths = [1, 3, 5]
for w in widths: 
    # apply spike jitter 
    shuffled_train_tuple = shuffle_spikes_in_time(train_truth, w)
    max_spike_count = max(np.max(train_truth), np.max(shuffled_train_tuple))

    # plot the difference in spikes
    fig, ax = plt.subplots(2, 1)
    c = 'plasma'
    im = ax[0].imshow(train_truth[0, :, :].T/max_spike_count, cmap=c, vmin=0, vmax=1)
    ax[1].imshow(shuffled_train_tuple[0, :, :].T/max_spike_count, cmap=c, vmin=0, vmax=1)
    ax[0].set_ylabel('Original Spikes')
    ax[1].set_ylabel('Jittered Spikes')
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.95)
    plt.suptitle('Width ' + str(w) + ' Spikes')
    plt.show(block=False)

    # check to make sure you don't lose spike mass over time 
    spikes_over_time = np.sum(train_truth[0, :, :], axis=0)
    jittered_spikes_over_time = np.sum(shuffled_train_tuple[0, :, :], axis=0)
    spikes_lost_over_time = np.unique(spikes_over_time - jittered_spikes_over_time)

    pdb.set_trace()
