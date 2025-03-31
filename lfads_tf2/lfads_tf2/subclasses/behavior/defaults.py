from lfads_tf2.defaults import get_cfg_defaults as get_base_cfg
from yacs.config import CfgNode as CN

def get_cfg_defaults():
    """Get default YACS config node for BehaviorLFADS"""
    _C = get_base_cfg()
    # Add subclass-specific hyperparameters
    _C.MODEL.ENC_INPUT_DIM = 50 # Dimensionality of the spike projection to feed into LFADS
    _C.MODEL.NORMALIZE = True # When true, initializes additional lowD readin to normalize data

    _C.TRAIN.FIX_READIN = False # Whether or not to freeze the lowD readin matrix weights
    _C.TRAIN.LR.DECODER = 1.e-3 # Learning rate to use for separate behavioral readout optimizer

    _C.TRAIN.DECODE = CN()
    _C.TRAIN.DECODE.FROM = 'gen_states' # Variable to predict from in behavioral mapping, should be gen_states or factors
    _C.TRAIN.DECODE.SCALE = 1.0 # Weight to apply to decoder NLL cost 
    _C.TRAIN.DECODE.START_EPOCH = 0
    _C.TRAIN.DECODE.INCREASE_EPOCH = 0
    _C.TRAIN.DECODE.N_DELAY_BINS = 0 # number of bins of delay to use in evaluating predictions

    return _C.clone()