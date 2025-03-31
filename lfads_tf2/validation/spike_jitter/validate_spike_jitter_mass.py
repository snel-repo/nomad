import h5py    
import pdb
import numpy as np    
from os import path
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
from lfads_tf2.utils import load_data
from lfads_tf2.tuples import BatchInput
from lfads_tf2.augmentations import shuffle_spikes_in_time
from lfads_tf2.defaults import get_cfg_defaults, DEFAULT_CONFIG_DIR

# ------------------------------------------------------------------

# set up the config for the lorenz data as a sample
cfg_path = path.join(DEFAULT_CONFIG_DIR, 'lorenz.yaml')
cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_path)
cfg.freeze()

# Load the spikes and the true rates
train_truth, valid_truth = load_data(
    cfg.TRAIN.DATA.DIR, 
    prefix=cfg.TRAIN.DATA.PREFIX, 
    signal='data')[0]

# make a fake BatchInput object 
# have to force the dataset to be one sample 
# since this is how map works 
train_ext = tf.zeros((train_truth.shape[1], 0))
train_sv_mask = tf.zeros(tf.shape(train_truth)[1:])

for i in np.random.randint(0, high=train_truth.shape[0], size=(4,)):
    train_data = tf.convert_to_tensor(train_truth[i, :, :])

    train_tuple = BatchInput(data=train_data, 
                            sv_mask=train_sv_mask, 
                            ext_input=train_ext)

    # apply spike jitter 
    shuffled_train_tuple = shuffle_spikes_in_time(train_tuple, 2)

    # look at the difference between the two datasets 
    difference = train_tuple.data - shuffled_train_tuple.data

    # compute max spike count 
    max_spike_count = tf.math.maximum(np.max(train_truth[i,:,:]), tf.math.reduce_max(shuffled_train_tuple.data)).numpy()

    # plot the difference in spikes
    fig, ax = plt.subplots(2, 1)
    c = 'plasma'
    im = ax[0].imshow(train_truth[i,:,:].T/max_spike_count, cmap=c, vmin=0, vmax=1)
    ax[1].imshow(tf.transpose(shuffled_train_tuple.data)/max_spike_count, cmap=c, vmin=0, vmax=1)
    ax[0].set_ylabel('Original Spikes')
    ax[1].set_ylabel('Jittered Spikes')
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.95)
    plt.suptitle('Sample ' + str(i) + ' Spikes')
    plt.show(block=False)

    # check to make sure you don't lose spike mass over time 
    spikes_over_time = np.sum(train_truth[i,:,:], axis=0)
    jittered_spikes_over_time = tf.math.reduce_sum(shuffled_train_tuple.data, axis=0)
    spikes_lost_over_time = np.unique(spikes_over_time - jittered_spikes_over_time)

    # plot any lost spike mass over time
    fig, ax = plt.subplots(3, 1)
    im = ax[0].imshow(np.expand_dims(spikes_over_time, axis=1).T, aspect=4)
    ax[0].set_ylabel('Spikes Over Time')
    ax[1].imshow(np.expand_dims(jittered_spikes_over_time.numpy(), axis=1).T, aspect=4)
    ax[1].set_ylabel('Jittered Spikes Over Time')
    ax[2].imshow(np.expand_dims((spikes_over_time - jittered_spikes_over_time).numpy(), axis=1).T, aspect=4)
    ax[2].set_ylabel('Difference')
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.95)
    plt.suptitle('Sample ' + str(i) + ' Spike Mass Over Time')
    plt.show(block=False)

pdb.set_trace()
