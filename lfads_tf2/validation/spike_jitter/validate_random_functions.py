import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import pdb

# script for comparing the random functions used in the LFL 
# and LF2 jitter functions 

# the widths to check
widths = [1, 3, 5]

# for each width
for w in widths: 
    # set the number of samples to draw
    nfound = 10000

    # draw the samples using TF2 function
    lf2 = tf.random.uniform([nfound], minval=-w, maxval=w, dtype=tf.dtypes.int32)
    # convert those samples to numpy to plot 
    lf2 = lf2.numpy()
    # draw the samples using the numpy function
    lfl = np.random.randint(-w, w, size=nfound)

    # plot overlapping histograms of samples drawn
    plt.figure()
    plt.hist(lfl, bins=(2*w), alpha=0.5)
    plt.hist(lf2, bins=(2*w), alpha=0.5)
    plt.legend(['Numpy', 'TF'])
    plt.show(block=False)

pdb.set_trace()