import pickle 
import numpy as np
import matplotlib.pyplot as plt 
import pdb

# the inputs to this script were obtained by putting breakpoints 
# in the respective LFADS code after spike jitter was applied 
# and pickling the jittered spikes. 
lfl_path = '/snel/home/brianna/lfl_jittered.pkl'
lf2_path = '/snel/home/brianna/lf2_jittered.pkl'

# load the LFL pickle file
infile = open(lfl_path, 'rb')
lfl = pickle.load(infile)
infile.close()

# load the LF2 pickle file
infile = open(lf2_path, 'rb')
lf2 = np.array(pickle.load(infile))
infile.close()

# set the max spike count 
max_spike_count = 8.0
# set the boundaries of the histogram bins
hist_bins = np.array(range(0, int(max_spike_count)+2)) - 0.5

# for each randomly selected sample
for i in np.random.randint(0, high=lfl.shape[0], size=(4,)):
    # select the sample from the pickled data 
    lfl_ = lfl[i, :, :]
    lf2_ = lf2[i, :, :]

    # plot the difference in spikes
    fig, ax = plt.subplots(2, 1)
    c = 'plasma'
    im = ax[0].imshow(lfl_.T/max_spike_count, cmap=c, vmin=0, vmax=1)
    ax[1].imshow(lf2_.T/max_spike_count, cmap=c, vmin=0, vmax=1)
    ax[0].set_ylabel('LFL Jittered Spikes')
    ax[1].set_ylabel('LF2 Jittered Spikes')
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.95)
    plt.suptitle('Sample ' + str(i) + ' Spikes')
    plt.show(block=False)

    # plot the distribution of spike counts in the non-zero region 
    plt.figure()
    plt.hist(lfl_[23:27, :].flatten(), bins=hist_bins, alpha=0.5, align='mid', ec='black')
    plt.hist(lf2_[23:27, :].flatten(), bins=hist_bins, alpha=0.5, align='mid', ec='black')
    plt.legend(['LFL', 'LF2'])
    plt.xlabel('Bin-Wise Spike Count')
    plt.ylabel('Frequency of Spike Count')
    plt.title('Sample ' + str(i) + ' Spike Count Frequency')
    plt.show(block=False)

# look at the distribution of spikes overall in the non-zero region
plt.figure()
plt.hist(lfl[23:27, :].flatten(), bins=hist_bins, alpha=0.5, align='mid', ec='black')
plt.hist(lf2[23:27, :].flatten(), bins=hist_bins, alpha=0.5, align='mid', ec='black')
plt.legend(['LFL', 'LF2'])
plt.xlabel('Bin-Wise Spike Count')
plt.ylabel('Frequency of Spike Count')
plt.title('All Samples - Spike Count Frequency')
plt.show(block=False)

pdb.set_trace()