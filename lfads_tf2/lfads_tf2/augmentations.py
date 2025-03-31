"""Functions for augmenting neural spiking data.

This module contains functions that modify spiking data immediately
prior to passing it into the model. They are best used in combination
with the `tf.Dataset.map` function.
"""

import numpy as np 
import tensorflow as tf 
from lfads_tf2.tuples import BatchInput


def shuffle_spikes_in_time(batch_data, w): 
    """
    A wrapper function around numpy_jitter that allows the numpy function to 
    be applied to tensors. Handles the application of jitter to a portion of 
    the BatchInput tuple via tf.map in models.py.

    Parameters
    ----------
    batch_data : lfads_tf2.tuples.BatchInput
        BatchInput object containing spike count data to be 
        shuffled as well as external inputs and sv_mask 
    w : float
        the maximum number of bins to shuffle spikes in either 
        direction

    Returns
    -------
    lfads_tf2.tuples.BatchInput
        BatchInput object containing shuffled spike data and 
        the unchanged external inputs and sv_mask.
    """

    # if jitter width is 0, then return the data unchanged
    if w == 0:
        return batch_data

    # define the jitter function
    jitter = lambda data: numpy_jitter(data, w)
    # turn the numpy impl into a tf function and apply to data
    shuffled_spikes = tf.numpy_function(jitter, [batch_data.data], tf.float32)
    
    # return a new BatchInput tuple with shuffled spikes
    return BatchInput(
        data=shuffled_spikes, 
        sv_mask=batch_data.sv_mask, 
        ext_input=batch_data.ext_input,
        dataset_name=batch_data.dataset_name,
        behavior=batch_data.behavior)


def numpy_jitter(data_bxtxd, w):
    """
    Shuffle the spikes in the temporal dimension.  This is useful to
    help the LFADS system avoid overfitting to individual spikes or fast
    oscillations found in the data that are irrelevant to behavior. A
    pure 'tabula rasa' approach would avoid this, but LFADS is sensitive
    enough to pick up dynamics that you may not want.

    NOTE: This function cannot be applied directly to a tf.Dataset 
    because it is implemented using `numpy`.

    Parameters
    ----------
    data_bxtxd : numpy.array
        Numpy array in the shape Time x Neurons containing the spiking 
        data to apply jitter to.
    w : int
        the maximum number of bins to shuffle spikes in either 
        direction

    Returns
    -------
    S_bxtxd : numpy.array
        Numpy array in the same shape as data_bxtxd but containing 
        the jittered spikes.
    """

    # get the shape of the data, which should be time by neurons
    T, N = data_bxtxd.shape

    # if passed width is 0, then return the data unchanged
    if w == 0:
        return data_bxtxd

    # determine the maximum spike count
    max_counts = int(np.max(data_bxtxd))
    # initialize an array to hold the shuffled spikes
    S_bxtxd = np.zeros([T,N])

    # Intuitively, shuffle spike occurances, 0 or 1, but since we have counts,
    # Do it over and over again up to the max count.
    for mc in range(1,max_counts+1):
        # pull out indices where there's a spike greater than mc
        idxs = np.nonzero(data_bxtxd >= mc)

        # initialize array where ones are found at indices
        # NOTE: the following 2 lines don't seem to be necessary
        data_ones = np.zeros_like(data_bxtxd)
        data_ones[data_bxtxd >= mc] = 1

        # the number of indices found
        nfound = len(idxs[0])
        # generate random shuffling increments between [-w, w]
        shuffles_incrs_in_time = np.random.randint(-w, w+1, size=nfound)

        # copy the indices
        shuffle_tidxs = idxs[0].copy()
        # add the shuffle to the indices
        shuffle_tidxs += shuffles_incrs_in_time

        # Reflect on the boundaries to not lose mass.
        # if the indices are negative, make them positive
        shuffle_tidxs[shuffle_tidxs < 0] = -shuffle_tidxs[shuffle_tidxs < 0]
        # if the indices are larger than the size of the array, move back within bounds
        shuffle_tidxs[shuffle_tidxs > T-1] = \
                                             (T-1)-(shuffle_tidxs[shuffle_tidxs > T-1] -(T-1))

        # for every new time index and the original neuron index
        for iii in zip(shuffle_tidxs, idxs[1]):
            # add a spike
            S_bxtxd[iii] += 1

    # return spikes as float 
    return S_bxtxd.astype(np.float32)


def shuffle_spikes_in_time_fast(batch_data, w):
    """DO NOT USE AS-IS
    
    A faster, but currently broken form of the spike jitter function.
    We haven't been able to implement the for-loop at the beginning in
    a form that can be traced with tf.function. A potential alternative
    is to vectorize this operation through something like the below 
    snippet (along with a few other downstream modifications), which 
    requires tf.repeat (only available in TF 2.1). 

    nz_indices = tf.where(data)
    counts = tf.gather_nd(data, nz_indices)
    all_indices = tf.repeat(nz_indices, counts, axis=0)
    
    We leave the implementation available in case we decide to fix it
    in the future.
    """

    if w == 0:
        return batch_data

    max_spike_ct = tf.reduce_max(data)
    all_indices = []
    for count in tf.range(1, max_spike_ct+1):
        indices = tf.where(data >= count)
        count_ixs = tf.cast(
            tf.fill([tf.shape(indices)[0], 1], count), tf.int64)
        all_indices.append(tf.concat([indices, count_ixs], axis=-1))
    all_indices = tf.concat(all_indices, axis=0)
    shifts = tf.random.uniform(
        [tf.shape(all_indices)[0]], 
        minval=-w, 
        maxval=w+1, # maxval is exclusive 
        dtype=tf.int64)
    # split up the indices to shift only the time index
    all_indices = tf.unstack(all_indices, axis=1)
    time_indices = all_indices[1]
    time_indices += shifts
    # reflect on the boundaries so we don't lose spikes
    B, T, N = tf.shape(data)
    # take care of negatives
    time_indices = tf.abs(time_indices) 
    # take care of positives
    oob_ixs = tf.where(time_indices > T-1)
    oob_values = tf.gather_nd(time_indices, oob_ixs)
    oob_values = 2*(T-1) - oob_values
    time_indices = tf.tensor_scatter_nd_update(
        time_indices, oob_ixs, oob_values)
    all_indices[1] = time_indices
    all_indices = tf.stack(all_indices, axis=-1)
    # create the summing tensor
    sum_tensor = tf.scatter_nd(
        all_indices, 
        tf.ones(tf.shape(all_indices)[0]), 
        [B, T, max_spike_ct])
    jittered_data = tf.reduce_sum(sum_tensor, axis=-1)

    return BatchInput(
        data=jittered_data, 
        sv_mask=batch_data.sv_mask, 
        ext_input=batch_data.ext_input)
