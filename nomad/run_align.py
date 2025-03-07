import os
import yaml
import matplotlib.pyplot as plt

from lfads_tf2.utils import load_data
from align_tf2.models import alignLFADS, Decoder
from align_tf2.defaults import get_cfg_defaults as get_align_defaults

# Read config to load data for evalution
cfg_path='/snel/home/asedler/core/align_tf2/config/align_jango.yaml'
cfg = get_align_defaults()
cfg.merge_from_file(cfg_path)
cfg.freeze()

# load some spikes for posterior sampling
chopped_spikes = load_data(cfg.TRAIN.DATA.DIR, merge_tv=True)

# create and train the aligner
model = alignLFADS(cfg_path=cfg_path)
model.train()

# perform posterior sampling on day 3 data
aligned_rates = model.sample_and_average(chopped_spikes[3], 3)

# plot and pause to examine the model
plt.plot(chopped_spikes[3][10,:,:])
plt.plot(aligned_rates[10,:,:])
plt.savefig(os.path.join(cfg.TRAIN.ALIGN_DIR, 'rates.png'))
import pdb; pdb.set_trace()