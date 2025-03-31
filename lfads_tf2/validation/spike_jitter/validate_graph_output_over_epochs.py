import pickle 
import glob
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt 
import pdb

# the inputs to this script were obtained by putting breakpoints 
# in the respective LFADS code after spike jitter was applied 
# and pickling the jittered spikes for multiple epochs, each
# in a separate file

# path to folder where all pickle files are found 
path = '/snel/home/brianna/projects/jitter/counts/'

# get the pickle files for lfl and lf2 in this folder 
lfl_paths = glob.glob(path + 'lfl*')
lf2_paths = glob.glob(path + 'lf2*')

# set max spike counts 
max_spike_count = 10.0
# set histogram bins
hist_bins = np.array(range(0, int(max_spike_count)+2)) - 0.5
# select a random sample to look at 
samples_to_search = np.random.randint(0, high=1248, size=(1,))

# for all of the epochs/files found
for i in range(0, len(lfl_paths)):
    # to handle different pickle protocols, use pandas function
    lfl = pd.read_pickle(lfl_paths[i])

    # load the lf2 data for that epoch
    infile = open(lf2_paths[i], 'rb')
    lf2 = np.array(pickle.load(infile))
    infile.close()

    # for the sample(s) selected, plot the spikes 
    for j in samples_to_search: 
        # plot the difference in spikes
        fig, ax = plt.subplots(2, 1)
        c = 'plasma'
        im = ax[0].imshow(lfl[j,:,:].T/max_spike_count, cmap=c, vmin=0, vmax=1)
        ax[1].imshow(lf2[j,:,:].T/max_spike_count, cmap=c, vmin=0, vmax=1)
        ax[0].set_ylabel('LFL Jittered Spikes')
        ax[1].set_ylabel('LF2 Jittered Spikes')
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.95)
        plt.suptitle('Epoch ' + str(i) + ', Sample ' + str(j))
        plt.show(block=False)

        # plot the histogram of spike counts in the non-zero region
        plt.figure()
        plt.hist(lfl[j, 23:27, :].flatten(), bins=hist_bins, alpha=0.5, align='mid', ec='black')
        plt.hist(lf2[j, 23:27, :].flatten(), bins=hist_bins, alpha=0.5, align='mid', ec='black')
        plt.legend(['LFL', 'LF2'])
        plt.xlabel('Bin-Wise Spike Count')
        plt.ylabel('Frequency of Spike Count')
        plt.title('Epoch ' + str(i) + ', Sample ' + str(j))
        plt.show(block=False)

    # look at the distribution of spikes overall 
    plt.figure()
    plt.hist(lfl[:, 23:27, :].flatten(), bins=hist_bins, alpha=0.5, align='mid', ec='black')
    plt.hist(lf2[:, 23:27, :].flatten(), bins=hist_bins, alpha=0.5, align='mid', ec='black')
    plt.legend(['LFL', 'LF2'])
    plt.xlabel('Bin-Wise Spike Count')
    plt.ylabel('Frequency of Spike Count')
    plt.title('All Samples - Spike Count Frequency, Epoch ' + str(i))
    plt.show(block=False)

pdb.set_trace()