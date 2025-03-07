from yacs.config import CfgNode as CN
from os import path

repo_path = path.dirname(path.realpath(__file__))
DEFAULT_CONFIG_DIR = path.join(repo_path, "config")

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100

# -----------------------------------------------------------------------------
# Model specs (Will typically vary between runs)
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'FFN-KL' 
_C.MODEL.DATA_DIM = 100
_C.MODEL.SEQ_LEN = 20
_C.MODEL.HIDDEN_DIM = 50
_C.MODEL.ALIGN_DATA = []
_C.MODEL.DAYK_LFADS_TRAINABLES = []
_C.MODEL.DAY0_MODEL_TYPE = 'dimreduced' # or 'behavior'
# posterior sampling 
_C.MODEL.SAMPLE_POSTERIORS = False
_C.MODEL.INIT_READIN_FROM_DAY0 = True # if true, initializes the dayk lowd readin from day0 model. if false initializes randomly

# -----------------------------------------------------------------------------
# Train Config
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.DATA = CN()
_C.TRAIN.DATA.DAY0 = "/path/to/day0/data"
_C.TRAIN.DATA.DAYK = "/path/to/dayk/data"

_C.TRAIN.PATIENCE = 40 # number of epochs of no improvement to val_loss before early stopping kicks in
_C.TRAIN.EARLY_STOPPING_COST = 'NLL-KL' #or 'NLL'
_C.TRAIN.BATCH_SIZE = 600 # number of elements in each batch for gradient descent
_C.TRAIN.VALID_BATCH_SIZE = -1 # batch size for validation, by default uses BATCH_SIZE
_C.TRAIN.MAX_EPOCHS = 10000 # the largest allowed number of training epochs
_C.TRAIN.ALIGN_LOSS_WEIGHT = [] # weights KL divergence for the FFN-KL and generator loss for the GAN
_C.TRAIN.LFADS_LOSS_WEIGHT = 1.0 # weights NLL + IC_KL + CO_KL + L2 on lowD readin
_C.TRAIN.MAX_GRAD_NORM = 200.0 # clips the global norm of the gradient
_C.TRAIN.LOSS_SCALE = 1.0 # a multiplier for the loss
_C.TRAIN.ADAM_EPSILON= 1.0e-8 # adam optimizer epsilon value

_C.TRAIN.GAN = CN()
_C.TRAIN.GAN.BATCH_DISC = True # whether to use the minibatch discrimination layer in the GAN's discriminator
_C.TRAIN.GAN.LABEL_SMOOTH = True # whether to use label smoothing for the discriminator
_C.TRAIN.GAN.DISC_TRAIN_DELAY = 20 # number of epochs to wait before training the discriminator
_C.TRAIN.GAN.DISC_LR = CN()
_C.TRAIN.GAN.DISC_LR.INIT = 0.001
_C.TRAIN.GAN.DISC_LR.STOP = 1.0e-10
_C.TRAIN.GAN.DISC_LR.DECAY = 1.0
_C.TRAIN.GAN.DISC_LR.PATIENCE = 6

_C.TRAIN.LR = CN()
_C.TRAIN.LR.INIT = 0.001
_C.TRAIN.LR.STOP = 1.0e-9
_C.TRAIN.LR.DECAY = 0.75
_C.TRAIN.LR.PATIENCE = 6

_C.TRAIN.L2 = CN()
_C.TRAIN.L2.READIN_SCALE = 1e-2  # the scale to weight the L2 of the low-dim

# KL cost is only applied when posterior sampling is turned on! 
_C.TRAIN.KL = CN()
_C.TRAIN.KL.START_EPOCH = 0 # the epoch during which to start ramping KL cost
_C.TRAIN.KL.INCREASE_EPOCH = 500 # the number of epochs to ramp KL cost
_C.TRAIN.KL.IC_WEIGHT = 1.0 # the scale to weight the KL of the IC's
_C.TRAIN.KL.CO_WEIGHT = 1.0 # the scale to weight the KL of the con outputs

_C.TRAIN.NLL = CN()
_C.TRAIN.NLL.START_EPOCH = 0 # the epoch during which to start ramping NLL cost
_C.TRAIN.NLL.INCREASE_EPOCH = 500 # the number of epochs to ramp NLL cost
_C.TRAIN.NLL.WEIGHT = 1.0 # the weight to scale reconstruction cost on rates 

# readin matrix
_C.TRAIN.FIX_READIN = False

_C.TRAIN.USE_TB = True
_C.TRAIN.NI_MODE = False # Whether or not this model is being managed by NomadInterface
_C.TRAIN.MODEL_DIR = "/path/to/day0/model"
_C.TRAIN.ALIGN_DIR = "/path/to/align/model"
_C.TRAIN.OVERWRITE = True

def get_cfg_defaults():
  """Get default LFADS config (yacs config node)."""
  return _C.clone()