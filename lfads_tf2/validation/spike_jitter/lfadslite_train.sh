#!/bin/bash

path_to_run_lfads=/snel/home/brianna/bin/lfadslite/run_lfadslite.py
if [ ! -n "$path_to_run_lfads" ]; then
    echo "Error: run_lfads.py not found on PATH. Ensure you add LFADS to your system PATH."
    exit 1
fi

DISPLAY=:21 CUDA_VISIBLE_DEVICES=0 python /snel/home/brianna/bin/lfadslite/run_lfadslite.py --data_dir=/snel/home/brianna/ --data_filename_stem=jitter --lfads_save_dir=/snel/home/brianna/lfads_output/ --temporal_spike_jitter_width=2 --keep_ratio=0.95 --ext_input_dim=0 --kl_increase_epochs=100 --l2_increase_epochs=100 --n_epochs_early_stop=0 --cell_clip_value=5.000000 --factors_dim=30 --in_factors_dim=0 --ic_enc_dim=100 --ci_enc_dim=100 --gen_dim=100 --keep_prob=1.0 --learning_rate_decay_factor=1 --device=/gpu:0 --co_dim=4 --do_causal_controller=false --do_feed_factors_to_controller=true --feedback_factors_or_rates=factors --controller_input_lag=1 --do_train_readin=false --l2_gen_scale=0.00217 --l2_con_scale=0.000123 --batch_size=50 --ic_dim=100 --con_dim=100 --learning_rate_stop=1e-10  --allow_gpu_growth=true --kl_ic_weight=7.086e-5 --kl_co_weight=6.518e-9 --inject_ext_input_to_gen=false 
