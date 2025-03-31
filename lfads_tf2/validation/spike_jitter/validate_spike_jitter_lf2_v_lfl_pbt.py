import os
import pickle
import pdb
import numpy as np

from rds.structures import * 
from rds.analyzer import *
from rds.decoding import * 
from alignment_analysis.run_scripts.nomad_plotting import plot_condition_grid
from textwrap import wrap

# ----------------
# params
# ----------------

# input rds object 
lfads_input = '/snel/share/share/derived/NW_Emory_Data_Sharing/Jango/2015/Jango_20150730_001/binsize_20ms/run_001/lfads_input/rds_Jango_20150730_001.pkl'
# tf2 pbt path
jitter_tf2_path = '/snel/home/brianna/ray_results/lfl_jitter_impl/best_model/posterior_samples.h5'
# lfl pbt path
jitter_lfl_path = '/snel/home/brianna/projects/blackbox/conditions/runs/Jango2015/lfads_output_0/16_workers_lfads_output/rds_Jango_20150730_001_1_0.pkl'

# smoothing width for empirical psths 
smoothing_width = 30
# condition separating field 
cond_sep_field='tgtDir'
# how to extract the data
extract_params = { 
        'calculate_params': {'name': 'moveOnset', 'threshold':0.15},
        'align_params': {'point':'backward_move_onset', 'window':(0.25,0.5)},
        'selection': {'result':'R'},
        'margins': 0.0
    }

# whether to plot psths, save psths, and where to save them
do_plot_psths = True
save_plots = True
figure_save_path = '/snel/home/brianna/projects/jitter/lfl_impl_psths/'
# make the save path if it doesn't exist 
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)

# ----------------
# load tf2 data
# ----------------

# load the rds object
fields = ['output_dist_params']
infile = open(lfads_input, 'rb')
tf2 = pickle.load(infile)
infile.close()

# load the lf2 posterior samples from the h5 file 
jitter_f = h5py.File(jitter_tf2_path, "r")
train_rates = jitter_f.get('train_rates')[:]
valid_rates = jitter_f.get('valid_rates')[:]
train_inds = jitter_f.get('train_inds')[:].astype(int)
valid_inds = jitter_f.get('valid_inds')[:].astype(int)

# reassemble the lfads rates
ntrials = train_inds.size + valid_inds.size
ndim = train_rates.shape[2]
nsamples = train_rates.shape[1]

model_rates = np.zeros((ntrials, nsamples, ndim))
model_rates[train_inds, :, :] = np.stack(train_rates)
model_rates[valid_inds, :, :] = np.stack(valid_rates)

# merge the lfads rates into the rds object 
tf2.merge_data_to_df(model_rates, 'lfads_rates', smooth_overlaps=True, overwrite=True)

# ----------------
# load lfl data
# ----------------

# load the lfl rds object 
# for this particular run, the rates are already merged back to the dataframe
infile = open(jitter_lfl_path, 'rb')
lfl = pickle.load(infile)
infile.close()

# ----------------
# get aligned data fields
# ----------------

# smooth the empirical spikes
smooth_spk_name = smooth_spikes(lfl, smoothing_width)

# get the lfads rates and smoothed spikes from each dataframe
tf2_cond_sep_data = get_aligned_data_fields(tf2, ['lfads_rates'], extract_params, cond_sep_field=cond_sep_field)
lfl_cond_sep_data = get_aligned_data_fields(lfl, ['lfads_rates', smooth_spk_name], extract_params, cond_sep_field=cond_sep_field)

# extract single trial and average data
tf2_avg_rates = tf2_cond_sep_data['lfads_rates'][0]
tf2_trial_rates = tf2_cond_sep_data['lfads_rates'][1]
lfl_avg_rates = lfl_cond_sep_data['lfads_rates'][0]
lfl_trial_rates = lfl_cond_sep_data['lfads_rates'][1]
trial_spikes = lfl_cond_sep_data[smooth_spk_name][1]

# ----------------
# compute empirical psths
# ----------------

# initialize array to hold empirical psths 
empirical_avg_rates = np.zeros(tf2_avg_rates.shape)

# take the average over each condition 
for ii, cond_data in enumerate(trial_spikes): 
    psth = np.mean(cond_data, axis = 2)
    empirical_avg_rates[:,:,ii] = psth

# ----------------
# zero out blank channels
# ----------------

# remove the blank channels from this dataset 
blank_channels = np.array([1, 10, 11, 15, 24, 30, 71, 91]) - 1

for bc in blank_channels: 
    tf2_avg_rates[:, bc, :] = 0
    lfl_avg_rates[:, bc, :] = 0 

    for cdata in tf2_trial_rates: 
        cdata[:, bc, :] = 0

    for cdata in lfl_trial_rates: 
        cdata[:, bc, :] = 0

    for cdata in trial_spikes: 
        cdata[:, bc, :] = 0

# ----------------
# compute R^2 between PSTHs 
# ----------------

# transpose data to make sure r2 is computed across neurons (not time)
empirical_psth = np.hstack(empirical_avg_rates).T
tf2_psth = np.hstack(tf2_avg_rates).T
lfl_psth = np.hstack(lfl_avg_rates).T

r2tf2 = r2(empirical_psth, tf2_psth)
r2lfl = r2(empirical_psth, lfl_psth)

print('Empirical vs. TF2: ' + str(np.mean(r2tf2)))
print('Empirical vs. LFL: ' + str(np.mean(r2lfl)))

# ----------------
# plot psths
# ----------------

if do_plot_psths: 
    cm = colormap.cool
    num_conditions = lfl_avg_rates.shape[2]
    # set range of neurons to be plotted per figure
    lower_bounds = np.linspace(0, 100, 11)
    # for each figure 
    for ii in range(0, len(lower_bounds)-1):
        lwbd = int(lower_bounds[ii])
        upbd = int(lower_bounds[ii+1])
        ylabels=[str(i) for i in range(lwbd, upbd)]
        # make a neurons x 3 subplot
        fig, ax = plt.subplots(upbd-lwbd, 3, sharex='col', sharey='row', figsize=(10,10))

        # plot the psths by condition 
        for c in range(0, num_conditions): 
            for n in range(lwbd, upbd): 
                ax[n-lwbd][0].plot(empirical_avg_rates[:, n, c], color=cm(c/float(num_conditions)))
                ax[n-lwbd][1].plot(tf2_avg_rates[:, n, c], color=cm(c/float(num_conditions)))
                ax[n-lwbd][2].plot(lfl_avg_rates[:, n, c], color=cm(c/float(num_conditions)))

                ax[n-lwbd][0].set_ylabel(n)
            
        ax[0][0].set_title('Empirical PSTH ' + str(smoothing_width) + ' ms')
        ax[0][1].set_title('TF2 PSTH')
        ax[0][2].set_title('LFL PSTH')
        plt.suptitle('TF2: %.3f, LFL: %.3f' % (np.mean(r2tf2), np.mean(r2lfl)))
        plt.show(block=False)
        plt.savefig(os.path.join(figure_save_path, 'psth_' + str(smoothing_width) + 'ms_' + str(ii) + '.png'))