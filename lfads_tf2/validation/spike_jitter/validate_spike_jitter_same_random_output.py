import numpy as np 
import tensorflow as tf 
from lfads_tf2.tuples import BatchInput
import matplotlib.pyplot as plt 
import pdb
# use python3/tf2 conda env for this script

# --------- random seeds ------------

# set random seeds for numpy and TF 
tf.random.set_seed(42)
np.random.seed(42)

# --------- make data ------------

# fake data to be jittered  
train_truth = np.zeros((50, 29))
train_ext = tf.zeros((train_truth.shape[0], 0))
train_sv_mask = tf.zeros(tf.shape(train_truth))

# assign one time point to have a high spike count 
train_truth[25, :] = 10.0 #np.random.randint(5, 10, size=(29,))

# make lf2 BatchInput object 
lf2_data = BatchInput(data=train_truth, 
                    sv_mask=train_sv_mask, 
                    ext_input=train_ext)

# set up LFL data 
lfl_data = np.expand_dims(train_truth, 0)

# jitter width
w = 2

# --------- apply lf2 jitter ------------
batch_data = lf2_data

# record the random shifts from this implementation to use later 
random_shifts = [] 

if w == 0:
    # return batch_data
    pass

spikes = batch_data.data
T = tf.shape(spikes)[0]
N = tf.shape(spikes)[1]

max_counts = tf.reduce_max(spikes, axis=None)
shuffled_spikes = tf.zeros([T, N], dtype=tf.dtypes.float32)

# Intuitively, shuffle spike occurances, 0 or 1, but since we have counts,
# Do it over and over again up to the max count.
for mc in tf.range(1, max_counts+1):
    idxs = tf.where(tf.not_equal(spikes >= mc, False)) # each index is a 2-item list [row, col]
    nfound = tf.shape(idxs)[0]

    # shuffle_incrs_in_time = tf.expand_dims(tf.random.uniform([nfound], minval=-w, maxval=w+1, dtype=tf.dtypes.int32), 1)
    shuffle_incrs_in_time = tf.expand_dims(tf.random.uniform([nfound], minval=-w, maxval=w, dtype=tf.dtypes.int32), 1)
    # store the random shift
    random_shifts.append(shuffle_incrs_in_time.numpy().squeeze())
    # shuffle_incrs_in_time = tf.expand_dims(tf.round(tf.random.truncated_normal([nfound], mean=0, stddev=w)), 1)
    # since we want to only change the time dim (row), add a column of 0s 
    shuffle_incrs_in_time = tf.concat([tf.cast(shuffle_incrs_in_time, tf.dtypes.int32), tf.zeros([nfound, 1], dtype=tf.dtypes.int32)], 1)
    shuffled_idxs = tf.cast(tf.identity(idxs), tf.dtypes.int32) + shuffle_incrs_in_time

    # Reflect on the boundaries to not lose mass.
    # if negative, flip the sign 
    shuffled_idxs = tf.abs(shuffled_idxs)

    # if greater than length of array, replace with (T-1) - (ind - (T-1))
    shuffled_idxs = tf.where(shuffled_idxs > T-1, x=(2*T-2) - shuffled_idxs, y=shuffled_idxs)

    # make a 2D tensor with a one at each index and add to existing spikes 
    shuffled_spikes = tf.tensor_scatter_nd_add(shuffled_spikes, shuffled_idxs, tf.ones(nfound, tf.float32))

lf2_output = shuffled_spikes.numpy()

# --------- apply lfl jitter ------------
data_bxtxd = lfl_data

B, T, N = data_bxtxd.shape
    #w = self.hps.temporal_spike_jitter_width

if w == 0:
    # return data_bxtxd
    pass

max_counts = int( np.max(data_bxtxd) )
S_bxtxd = np.zeros([B,T,N])

# Intuitively, shuffle spike occurances, 0 or 1, but since we have counts,
# Do it over and over again up to the max count.
for mc in range(1,max_counts+1):
    idxs = np.nonzero(data_bxtxd >= mc)

    data_ones = np.zeros_like(data_bxtxd)
    data_ones[data_bxtxd >= mc] = 1

    nfound = len(idxs[0])
    # pull the appropriate random shift from the store
    shuffles_incrs_in_time = random_shifts[mc-1]
    shuffle_tidxs = idxs[1].copy()
    shuffle_tidxs += shuffles_incrs_in_time

    # Reflect on the boundaries to not lose mass.
    shuffle_tidxs[shuffle_tidxs < 0] = -shuffle_tidxs[shuffle_tidxs < 0]
    shuffle_tidxs[shuffle_tidxs > T-1] = \
                                            (T-1)-(shuffle_tidxs[shuffle_tidxs > T-1] -(T-1))

    for iii in zip(idxs[0], shuffle_tidxs, idxs[2]):
        S_bxtxd[iii] += 1

lfl_output = S_bxtxd

# --------- plot ------------

# find max spike counts from each input/output
max_spike_count_lfl = max(np.max(train_truth), np.max(lfl_output))
max_spike_count_lf2 = max(np.max(train_truth), np.max(lf2_output))

# plot the difference in spikes
fig, ax = plt.subplots(2, 2)
c = 'plasma'
im = ax[0][0].imshow(lfl_data[0, :, :].T/max_spike_count_lfl, cmap=c, vmin=0, vmax=1)
ax[1][0].imshow(lfl_output[0, :, :].T/max_spike_count_lfl, cmap=c, vmin=0, vmax=1)
ax[0][0].set_ylabel('Original Spikes')
ax[1][0].set_ylabel('LFL Jittered Spikes')

im = ax[0][1].imshow(lf2_data.data.T/max_spike_count_lf2, cmap=c, vmin=0, vmax=1)
ax[1][1].imshow(lf2_output.T/max_spike_count_lf2, cmap=c, vmin=0, vmax=1)
ax[0][1].set_ylabel('Original Spikes')
ax[1][1].set_ylabel('LF2 Jittered Spikes')

cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.95)
plt.suptitle('Width ' + str(w) + ' Spikes')
plt.show(block=False)

pdb.set_trace()