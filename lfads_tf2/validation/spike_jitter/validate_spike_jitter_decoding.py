import os
import pickle
import pdb
import numpy as np

from rds.structures import * 
from rds.analyzer import *
from rds.decoding import * 
from alignment_analysis.run_scripts.nomad_plotting import plot_condition_grid, WF_decode_train, get_wrist_emg
from textwrap import wrap

# ----------------
# params
# ----------------

# path to rds pickle file 
lfads_input = '/snel/share/share/derived/NW_Emory_Data_Sharing/Jango/2015/Jango_20150730_001/binsize_20ms/run_001/lfads_input/rds_Jango_20150730_001.pkl'
# path to tf2 psths 
jitter_tf2_path = '/snel/home/brianna/ray_results/lfl_jitter_impl/best_model/posterior_samples.h5'
# path to lfl psths (this run already has them merged back to rds)
jitter_lfl_path = '/snel/home/brianna/projects/blackbox/conditions/runs/Jango2015/lfads_output_0/16_workers_lfads_output/rds_Jango_20150730_001_1_0.pkl'

# set smoothing width for empirical psths
smoothing_width = 60
# condition separating field
cond_sep_field='tgtDir'
# params to extract and align data for decoding 
extract_params = { 
        'calculate_params': {'name': 'moveOnset', 'threshold':0.15},
        'align_params': {'point':'backward_move_onset', 'window':(0.25,0.5)},
        'selection': {'result':'R'},
        'margins': 0.0
    }

# where and whether to plot 
do_plot_psths = True
save_plots = True
figure_save_path = '/snel/home/brianna/projects/jitter/lfl_impl_psths/'
# make savedir if it doesn't exist 
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

# load tf2 rate h5 file 
jitter_f = h5py.File(jitter_tf2_path, "r")
train_rates = jitter_f.get('train_rates')[:]
valid_rates = jitter_f.get('valid_rates')[:]
train_inds = jitter_f.get('train_inds')[:].astype(int)
valid_inds = jitter_f.get('valid_inds')[:].astype(int)

# reconstruct the lfads rates
ntrials = train_inds.size + valid_inds.size
ndim = train_rates.shape[2]
nsamples = train_rates.shape[1]

model_rates = np.zeros((ntrials, nsamples, ndim))
model_rates[train_inds, :, :] = np.stack(train_rates)
model_rates[valid_inds, :, :] = np.stack(valid_rates)

# merge them back to the df
tf2.merge_data_to_df(model_rates, 'lfads_rates', smooth_overlaps=True, overwrite=True)

# ----------------
# load lfl data
# ----------------

# load the pickle file with the lfl rds and rates already added 
infile = open(jitter_lfl_path, 'rb')
lfl = pickle.load(infile)
infile.close()

# ----------------
# get aligned data fields
# ----------------

# get smoothed empirical spikes 
smooth_spk_name = smooth_spikes(lfl, smoothing_width)

# extract condition separated aligned data 
tf2_cond_sep_data = get_aligned_data_fields(tf2, ['lfads_rates'], extract_params, cond_sep_field=cond_sep_field)
lfl_cond_sep_data = get_aligned_data_fields(lfl, ['lfads_rates', smooth_spk_name, 'force', 'raw_emg'], extract_params, cond_sep_field=cond_sep_field)

# get single trial and trial averaged data 
tf2_avg_rates = tf2_cond_sep_data['lfads_rates'][0]
tf2_trial_rates = tf2_cond_sep_data['lfads_rates'][1]
lfl_avg_rates = lfl_cond_sep_data['lfads_rates'][0]
lfl_trial_rates = lfl_cond_sep_data['lfads_rates'][1]
trial_spikes = lfl_cond_sep_data[smooth_spk_name][1]
trial_force = lfl_cond_sep_data['force'][1]
trial_emg = lfl_cond_sep_data['raw_emg'][1]

# extract wrist muscles from emg data only 
trial_emg, wrist_emg_labels = get_wrist_emg(lfl, trial_emg)

# ----------------
# compute empirical psths
# ----------------

# initialize array to hold empirical psths 
empirical_avg_rates = np.zeros(tf2_avg_rates.shape)

# compute average over each condition
for ii, cond_data in enumerate(trial_spikes): 
    psth = np.mean(cond_data, axis = 2)
    empirical_avg_rates[:,:,ii] = psth

# ----------------
# decode force 
# ----------------

# train and evaluate decoder on lfl rates
lfl_vaf_force, lfl_wf_force, lfl_wf_force_preds, lfl_wf_weights_force = \
    WF_decode_train(lfl_trial_rates, trial_force, n_history=3, metric='vaf')

# plot lfl force decoding
t = 'LFL, VAF: ' + str(lfl_vaf_force)
fig, ax = plot_condition_grid(lfl_wf_force, extract_params['align_params'], label='True')
fig, ax = plot_condition_grid(lfl_wf_force_preds, extract_params['align_params'], fig_ax=[fig, ax], \
    color='r', title=t, legend=True, label='Decoded', to_save=save_plots, save_path=figure_save_path, \
        save_name='lfl_decoded_force')

# train and evaluate decoder on tf2 rates
tf2_vaf_force, tf2_wf_force, tf2_wf_force_preds,tf2_wf_weights_force = \
    WF_decode_train(tf2_trial_rates, trial_force, n_history=3, metric='vaf')

# plot lf2 force decoding 
t = 'TF2, VAF: ' + str(tf2_vaf_force)
fig, ax = plot_condition_grid(tf2_wf_force, extract_params['align_params'], label='True')
fig, ax = plot_condition_grid(tf2_wf_force_preds, extract_params['align_params'], fig_ax=[fig, ax], \
    color='r', title=t, legend=True, label='Decoded', to_save=save_plots, save_path=figure_save_path, \
        save_name='tf2_decoded_force')

# ----------------
# decode emg 
# ----------------

# train and evaluate decoder on lfl rates
lfl_vaf_emg, lfl_wf_emg, lfl_wf_emg_preds, lfl_wf_weights_emg = \
    WF_decode_train(lfl_trial_rates, trial_emg, n_history=3, metric='vaf')

# plot lfl emg decoding 
t = 'LFL, VAF: ' + str(lfl_vaf_emg)
fig, ax = plot_condition_grid(lfl_wf_emg, extract_params['align_params'], label='True', ylabels=wrist_emg_labels)
fig, ax = plot_condition_grid(lfl_wf_emg_preds, extract_params['align_params'], fig_ax=[fig, ax], \
    color='r', title=t, legend=True, label='Decoded', to_save=save_plots, save_path=figure_save_path, \
        save_name='lfl_decoded_emg', ylabels=wrist_emg_labels)

# train and evaluate decoder on tf2 rates
tf2_vaf_emg, tf2_wf_emg, tf2_wf_emg_preds, tf2_wf_weights_emg = \
    WF_decode_train(tf2_trial_rates, trial_emg, n_history=3, metric='vaf')

# plot lf2 emg decoding 
t = 'TF2, VAF: ' + str(tf2_vaf_emg)
fig, ax = plot_condition_grid(tf2_wf_emg, extract_params['align_params'], label='True', ylabels=wrist_emg_labels)
fig, ax = plot_condition_grid(tf2_wf_emg_preds, extract_params['align_params'], fig_ax=[fig, ax], \
    color='r', title=t, legend=True, label='Decoded', to_save=save_plots, save_path=figure_save_path, \
        save_name='tf2_decoded_emg', ylabels=wrist_emg_labels)