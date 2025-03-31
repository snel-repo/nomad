import h5py    
import pdb
import numpy as np    
from os import path
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
from lfads_tf2.tuples import BatchInput
from lfads_tf2.augmentations import shuffle_spikes_in_time
tf.random.set_seed(42)
np.random.seed(42)

# make a fake dataset 
# have to force the dataset to be one sample 
# since this is how map works 
train_truth = np.zeros((50, 29))
train_ext = tf.zeros((train_truth.shape[0], 0))
train_sv_mask = tf.zeros(tf.shape(train_truth))

# assign one time point to have a high spike count 
train_truth[0, :] = 8.0
train_truth = tf.convert_to_tensor(train_truth)

# make a fake dataset 
# have to force the dataset to be one sample 
# since this is how map works 
train_tuple = BatchInput(data=train_truth, 
                        sv_mask=train_sv_mask, 
                        ext_input=train_ext)

# run spike jitter with different widths 
widths = [1, 3, 5]
for w in widths: 
    # apply spike jitter 
    shuffled_train_tuple = shuffle_spikes_in_time(train_tuple, w)
    # compute max spike count for input and output
    max_spike_count = tf.math.maximum(np.max(train_tuple.data), tf.math.reduce_max(shuffled_train_tuple.data)).numpy()

    # plot the difference in spikes
    fig, ax = plt.subplots(2, 1)
    c = 'plasma'
    im = ax[0].imshow(train_tuple.data.numpy().T/max_spike_count, cmap=c, vmin=0, vmax=1)
    ax[1].imshow(tf.transpose(shuffled_train_tuple.data)/max_spike_count, cmap=c, vmin=0, vmax=1)
    ax[0].set_ylabel('Original Spikes')
    ax[1].set_ylabel('Jittered Spikes')
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.95)
    plt.suptitle('Width ' + str(w) + ' Spikes')
    plt.show(block=False)

    # check to make sure you don't lose spike mass over time 
    spikes_over_time = np.sum(train_truth, axis=0)
    jittered_spikes_over_time = np.sum(shuffled_train_tuple.data, axis=0)
    spikes_lost_over_time = np.unique(spikes_over_time - jittered_spikes_over_time)

pdb.set_trace()
