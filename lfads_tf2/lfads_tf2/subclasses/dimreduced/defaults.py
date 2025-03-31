from lfads_tf2.defaults import get_cfg_defaults as get_base_cfg

def get_cfg_defaults():
    """Get default YACS config node for DimReducedLFADS"""
    _C = get_base_cfg()
    # Add subclass-specific hyperparameters
    _C.MODEL.ENC_INPUT_DIM = 50 # Dimensionality of the spike projection to feed into LFADS
    _C.MODEL.NORMALIZE = True # When true, initializes lowD readin to normalize data
    _C.TRAIN.FIX_READIN = False # Whether or not to freeze the lowD readin matrix weights

    return _C.clone()
