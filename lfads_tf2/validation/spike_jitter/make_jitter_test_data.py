import h5py    
import pdb
import pickle 
from rds.structures import Dataset
import numpy as np   

# the path to the original data 
# this is just used to get the shape the data should be 
# allows us to avoid re-chopping the data ourselves
datapath = '/snel/share/data/lfads_lorenz_20ms/lfads_dataset001.h5'
# path to save the new datafile 
savepath = '/snel/home/brianna/jitter_spike_count.h5'

# open and read the train/valid data and inds
jitter_f = h5py.File(datapath, "r")

train = jitter_f.get('train_data')[:]
valid = jitter_f.get('valid_data')[:]
train_inds = jitter_f.get('train_inds')[:]
valid_inds = jitter_f.get('valid_inds')[:]

# initialize arrays to hold the jitter test data
jitter_train = np.zeros(train.shape)
jitter_valid = np.zeros(valid.shape)

# for each training sample, set up your desired jitter data
for jt in jitter_train: 
    jt[25, :] = np.random.randint(5, 10, size=(29,))

# for each validation sample, set up your desired jitter data
for jv in jitter_valid: 
    jv[25, :] = np.random.randint(5, 10, size=(29,))

# save the new train/valid data and the original inds
with h5py.File(savepath, 'w') as h5f:
    h5f.create_dataset('train_data', data=jitter_train, compression='gzip')
    h5f.create_dataset('valid_data', data=jitter_valid, compression='gzip')
    h5f.create_dataset('train_inds', data=train_inds, compression='gzip')
    h5f.create_dataset('valid_inds', data=valid_inds, compression='gzip')
print('Sucessfully wrote the data to %s' % savepath)
