from lfads_tf2.utils import restrict_gpu_usage
restrict_gpu_usage()
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import Progbar
from tensorflow.keras import Model

import tensorflow_probability as tfp
tfd = tfp.distributions

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging.config
import numpy as np
import pandas as pd
import git, shutil, h5py, yaml, copy, sys, os, io
from os import path

from lfads_tf2.defaults import get_cfg_defaults
from lfads_tf2.utils import load_data, flatten, load_posterior_averages
from lfads_tf2.tuples import LoadableData, BatchInput, DecoderInput, \
    LFADSInput, LFADSOutput, SamplingOutput
from lfads_tf2.layers import Encoder, Decoder, AutoregressiveMultivariateNormal
from lfads_tf2.initializers import variance_scaling
from lfads_tf2.augmentations import shuffle_spikes_in_time
from lfads_tf2.regularizers import DynamicL2


class LFADS(Model):
    """
    Defines the LFADS model, a sequential autoencoder which takes as 
    input a batch of time segments of binned neural spikes, computes 
    a distributions over initial conditions for a generator network and
    a sequence of inputs to a controller network and then uses them 
    to generate a sequence of controller output posterior distributions, 
    generator states, factors, and estimated Poisson rates for the observed
    neurons. Note that `Decoder` is a subclass of `tensorflow.keras.Model`.
    """
    def __init__(self, cfg_node=None, cfg_path=None, model_dir=None):
        """ Initializes an LFADS object.

        This method will create a new model based on a specified configuration
        and create a `model_dir` in which to store all model-related data. 

        Parameters
        ----------
        cfg_node: yacs.config.CfgNode, optional
            yacs CfgNode to override the default CfgNode
        cfg_path: str, optional
            A path to a YAML update for the default YACS config node. 
            If not provided, use the defaults.
        model_dir: str, optional
            If provided, an LFADS run directory.
        """

        super(LFADS, self).__init__()
        # Build the model and perform housekeeping
        self.build_model_from_init_call(
            cfg_node=cfg_node, cfg_path=cfg_path, model_dir=model_dir)
        # Restore weights
        if self.from_existing:
            self.restore_weights(lve=False)
    
    def get_cfg_defaults(self):
        """Returns the LFADS configuration defaults.

        NOTE: Overloadable

        Returns
        -------
        yacs.config.CfgNode
            The default configuration for the model.
        """
        return get_cfg_defaults()
    
    def get_input_shapes(self, batch_size=None):
        """Returns the tensor shapes that TF should expect as input to
        the compute graph.

        NOTE: Overloadable
        """
        mcfg = self.cfg.MODEL
        data_shape = [batch_size, mcfg.SEQ_LEN, mcfg.DATA_DIM]
        enc_input_shape = [batch_size, mcfg.SEQ_LEN, mcfg.ENC_INPUT_DIM]
        ext_input_shape = [batch_size, mcfg.SEQ_LEN, mcfg.EXT_INPUT_DIM]
        name_shape = [batch_size,]
        behavior_shape = [batch_size, mcfg.SEQ_LEN, mcfg.BEHAVIOR_DIM] # shouldn't matter since this isn't used for anything in main LFADS
        return data_shape, enc_input_shape, ext_input_shape, name_shape, behavior_shape

    def build_model_from_init_call(self,
                                   cfg_node=None,
                                   cfg_path=None,
                                   model_dir=None):
        """Builds the LFADS model using a configuration.

        This does most of the work of building the model on a call to 
        __init__. It has been abstracted from the __init__ function 
        to allow for easier subclassing without having to worry about
        the weight restoration step.

        NOTE: Overloadable

        Parameters
        ----------
        cfg_node: yacs.config.CfgNode, optional
            yacs CfgNode to override the default CfgNode
        cfg_path: str, optional
            A path to a YAML update for the default YACS config node. 
            If not provided, use the defaults.
        model_dir: str, optional
            If provided, an LFADS run directory.
        """

        # Check args
        num_set = sum([x != None for x in [cfg_node, cfg_path, model_dir]])
        if num_set > 1:
            raise ValueError(
                "Only one of `cfg_node`, `cfg_path`, or `model_dir` may be set")

        # Get the commit information for this lfads_tf2 code
        repo_path = path.realpath(
            path.join(path.dirname(__file__), '..'))
        repo = git.Repo(path=repo_path)
        git_data = {
            'commit': repo.head.object.hexsha,
            'modified': [diff.b_path for diff in repo.index.diff(None)],
            'untracked': repo.untracked_files,
        }

        if model_dir: # Load existing model - model_dir should contain its own cfg
            print("Loading model from {}.".format(model_dir))
            assert path.exists(model_dir), \
                "No model exists at {}".format(model_dir)
            ms_path = path.join(model_dir, 'model_spec.yaml')

            cfg = self.get_cfg_defaults()
            cfg.merge_from_file(ms_path)
            # check that the model directory is correct
            if not cfg.TRAIN.MODEL_DIR == model_dir:
                print("The `model_dir` in the config file doesn't match "
                "the true model directory. Updating and saving new config.")
                cfg.TRAIN.MODEL_DIR = model_dir
                ms_path = path.join(model_dir, 'model_spec.yaml')
                with open(ms_path, 'w') as ms_file:
                    ms_file.write(cfg.dump())
            cfg.freeze()

            # check that code used to train the model matches this code
            git_data_path = path.join(model_dir, 'git_data.yaml')
            trained_git_data = yaml.full_load(open(git_data_path))
            if trained_git_data != git_data:
                print(
                    'This `lfads_tf2` may not match the one '
                    'used to create the model.'
                )
            self.is_trained = True
            self.from_existing = True
        else: # Fresh model
            # Read config and prepare model directory
            print("Initializing new model.")
            cfg = self.get_cfg_defaults()
            if cfg_node:
                cfg.merge_from_other_cfg(cfg_node)
            elif cfg_path:
                cfg.merge_from_file(cfg_path)
            else:
                print("WARNING - Using default config")
            cfg.TRAIN.DATA.DIR = path.expanduser(cfg.TRAIN.DATA.DIR)
            cfg.TRAIN.MODEL_DIR = path.expanduser(cfg.TRAIN.MODEL_DIR)
            cfg.freeze()

            # Ensure that the model directory is handled appropriately
            model_dir = cfg.TRAIN.MODEL_DIR
            if not cfg.TRAIN.TUNE_MODE:
                if cfg.TRAIN.OVERWRITE:
                    if path.exists(model_dir):
                        print("Overwriting model directory...")
                        shutil.rmtree(model_dir)
                else:
                    assert not path.exists(model_dir), \
                        (f"A model already exists at `{model_dir}`. "
                        "Load it by explicitly providing the path, or specify a new `model_dir`.")
            # Create model directory
            os.makedirs(model_dir, exist_ok=True)

            # Save the model_spec
            try:
                ms_path = path.join(model_dir, 'model_spec.yaml')
                with open(ms_path, 'w') as ms_file:
                    ms_file.write(cfg.dump())
            except FileNotFoundError:
                print("File error. Please check that TUNE_MODE is set correctly.")
                raise

            # Save the git commit information
            git_data_path = path.join(model_dir, 'git_data.yaml')
            with open(git_data_path, 'w') as git_data_file:
                yaml.dump(git_data, git_data_file)

            # Create the model from the config file - must happen after model directory handling
            # because creation of the SummaryWriter automatically creates the model directory.
            self.is_trained = False
            self.from_existing = False

        # ==================================
        # ===== SET UP THE LFADS MODEL =====
        # ==================================

        # ===== SET UP THE LOGGERS =====
        # set up logging using the configuration file in this directory
        log_conf_path = path.join(path.dirname(__file__), 'logging_conf.yaml')
        logging_conf = yaml.full_load(open(log_conf_path))
        # in `TUNE_MODE`, restrict the console handler to WARNING's only
        if cfg.TRAIN.TUNE_MODE:
            logging_conf['handlers']['console']['level'] = 'WARNING'
            print('TUNE_MODE is True, so only warnings will be logged.\n'
                  'To see the progress bar and other info, set TUNE_MODE to '
                  'False.')
        # tell the file handlers where to write output
        logging_conf['handlers']['logfile']['filename'] = path.join(
            cfg.TRAIN.MODEL_DIR, 'train.log')
        logging_conf['handlers']['csv']['filename'] = path.join(
            cfg.TRAIN.MODEL_DIR, 'train_data.csv')
        # set up logging from the configuation dict and create a global logger
        logging.config.dictConfig(logging_conf)
        self.lgr = logging.getLogger('lfads')
        self.csv_lgr = logging.getLogger('train_csv')
        # Prevent duplicate messages from another logging configuration
        self.lgr.propagate = False
        self.csv_lgr.propagate = False

        # ===== SETTING ENCODER INPUT SIZE =====
        if type(self) == LFADS:
            # Ensure that encoder input is same dim as spikes
            data_dim = cfg.MODEL.DATA_DIM
            if cfg.MODEL.ENC_INPUT_DIM != data_dim:
                self.lgr.warn("For LFADS, ENC_INPUT_DIM must equal "
                    f"DATA_DIM. Setting the former to {data_dim}")
                cfg.defrost()
                cfg.MODEL.ENC_INPUT_DIM = data_dim
                cfg.freeze()

        # ===== DECISION TO USE CONTROLLER =====
        self.use_con = all([
            cfg.MODEL.CI_ENC_DIM > 0,
            cfg.MODEL.CON_DIM > 0,
            cfg.MODEL.CO_DIM > 0])
        if not self.use_con:
            self.lgr.info("A controller-related dim was set to zero. "
                "Turning off all controller-related HPs.")
            cfg.defrost()
            cfg.MODEL.CI_ENC_DIM = 0
            cfg.MODEL.CON_DIM = 0
            cfg.MODEL.CO_DIM = 0
            cfg.TRAIN.L2.CI_ENC_SCALE = 0.0
            cfg.TRAIN.L2.CON_SCALE = 0.0
            cfg.TRAIN.KL.CO_WEIGHT = 0.0
            cfg.freeze()

        # ===== DECISION TO USE RAMPING =====
        self.use_kl_ramping = any([
            cfg.TRAIN.KL.IC_WEIGHT,
            cfg.TRAIN.KL.CO_WEIGHT])
        self.use_l2_ramping = any([
            cfg.TRAIN.L2.IC_ENC_SCALE,
            cfg.TRAIN.L2.GEN_SCALE,
            cfg.TRAIN.L2.CI_ENC_SCALE,
            cfg.TRAIN.L2.CON_SCALE])
        cfg.defrost()
        if not self.use_kl_ramping:
            self.lgr.info("No KL weights found. Turning off KL ramping.")
            cfg.TRAIN.KL.START_EPOCH = 0
            cfg.TRAIN.KL.INCREASE_EPOCH = 0
        if not self.use_l2_ramping:
            self.lgr.info("No L2 weights found. Turning off L2 ramping.")
            cfg.TRAIN.L2.START_EPOCH = 0
            cfg.TRAIN.L2.INCREASE_EPOCH = 0
        cfg.freeze()

        # ===== INITIALIZE CONFIG VARIABLES =====
        # create variables for L2 and KL ramping weights
        self.kl_ramping_weight = tf.Variable(0.0, trainable=False)
        self.l2_ramping_weight = tf.Variable(0.0, trainable=False)
        # create variables for dropout rates
        self.dropout_rate = tf.Variable(cfg.MODEL.DROPOUT_RATE, trainable=False)
        self.cd_keep = tf.Variable(1 - cfg.MODEL.CD_RATE, trainable=False)
        self.cd_pass_rate = tf.Variable(cfg.MODEL.CD_PASS_RATE, trainable=False)
        self.sv_keep = tf.Variable(1 - cfg.MODEL.SV_RATE, trainable=False)
        # create variables for loss and gradient modfiers
        self.max_grad_norm = tf.Variable(cfg.TRAIN.MAX_GRAD_NORM, trainable=False)
        self.loss_scale = tf.Variable(cfg.TRAIN.LOSS_SCALE, trainable=False)
        # create variable for learning rate
        self.learning_rate = tf.Variable(cfg.TRAIN.LR.INIT, trainable=False)
        # create a variable that indicates whether the model is training
        self.training = tf.Variable(False, trainable=False)
        # compute total recurrent size of the model for L2 regularization
        def compute_recurrent_size(cfg):
            t_cfg, m_cfg = cfg.TRAIN, cfg.MODEL
            recurrent_units_and_weights = [
                (m_cfg.IC_ENC_DIM, t_cfg.L2.IC_ENC_SCALE),
                (m_cfg.IC_ENC_DIM, t_cfg.L2.IC_ENC_SCALE),
                (m_cfg.CI_ENC_DIM, t_cfg.L2.CI_ENC_SCALE),
                (m_cfg.CI_ENC_DIM, t_cfg.L2.CI_ENC_SCALE),
                (m_cfg.GEN_DIM, t_cfg.L2.GEN_SCALE),
                (m_cfg.CON_DIM, t_cfg.L2.CON_SCALE)]
            model_recurrent_size = 0
            for units, weight in recurrent_units_and_weights:
                if weight > 0:
                    model_recurrent_size += 3 * units**2
            return model_recurrent_size
        # total size of all recurrent kernels - note there are three gates to calculate
        self.model_recurrent_size = compute_recurrent_size(cfg)

        # ===== CREATE THE PRIORS =====
        with tf.name_scope('priors'):
            # create the IC prior variables
            self.ic_prior_mean = tf.Variable(
                tf.zeros(cfg.MODEL.IC_DIM),
                name='ic_prior_mean')
            self.ic_prior_logvar = tf.Variable(
                tf.fill([cfg.MODEL.IC_DIM],
                    tf.math.log(cfg.MODEL.IC_PRIOR_VAR)),
                trainable=False,
                name='ic_prior_logvar')
            # create the CO prior variables
            trainable_decoder = not cfg.TRAIN.ENCODERS_ONLY
            self.logtaus = tf.Variable(
                tf.fill([cfg.MODEL.CO_DIM],
                    tf.math.log(cfg.MODEL.CO_PRIOR_TAU)),
                trainable=self.use_con & trainable_decoder,
                name='logtaus')
            self.lognvars = tf.Variable(
                tf.fill([cfg.MODEL.CO_DIM],
                    tf.math.log(cfg.MODEL.CO_PRIOR_NVAR)),
                trainable=self.use_con & trainable_decoder,
                name='lognvars')
        # create the autoregressive prior distribution
        self.co_prior = AutoregressiveMultivariateNormal(
            self.logtaus, self.lognvars, cfg.MODEL.CO_DIM, name='co_prior')
        # create the KL weight variables
        self.kl_ic_weight = tf.Variable(
            cfg.TRAIN.KL.IC_WEIGHT, trainable=False, name='kl_ic_weight')
        self.kl_co_weight = tf.Variable(
            cfg.TRAIN.KL.CO_WEIGHT, trainable=False, name='kl_co_weight')

        # ===== CREATE THE ENCODER AND DECODER =====
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        if cfg.TRAIN.ENCODERS_ONLY:
            # Turn off decoder training
            self.lgr.warn('Training encoder only.')
            self.decoder.trainable = False

        # ===== CREATE THE READOUT MATRIX =====
        # create the mapping from factors to rates
        self.rate_linear = Dense(cfg.MODEL.DATA_DIM,
            kernel_initializer=variance_scaling,
            name='rate_linear'
        )

        # ===== CREATE THE OPTIMIZER =====
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            epsilon=cfg.TRAIN.ADAM_EPSILON)

        # ===== CREATE THE CHECKPOINTS =====
        ckpt_dir = path.join(cfg.TRAIN.MODEL_DIR, 'lfads_ckpts')
        # checkpointing for least validation error model
        self.lve_ckpt = tf.train.Checkpoint(model=self)
        lve_ckpt_dir = path.join(ckpt_dir, 'least_val_err')
        self.lve_manager = tf.train.CheckpointManager(
                self.lve_ckpt, directory=lve_ckpt_dir,
                max_to_keep=1, checkpoint_name='lve-ckpt')
        # checkpointing for the most recent model
        self.mrc_ckpt = tf.train.Checkpoint(model=self)
        mrc_ckpt_dir = path.join(ckpt_dir, 'most_recent')
        self.mrc_manager = tf.train.CheckpointManager(
            self.mrc_ckpt, directory=mrc_ckpt_dir,
            max_to_keep=1, checkpoint_name='mrc-ckpt')

        # ===== CREATE TensorBoard SUMMARY_WRITER =====
        if cfg.TRAIN.USE_TB:
            self.summary_writer = tf.summary.create_file_writer(cfg.TRAIN.MODEL_DIR)

        # ===== CREATE TRAINING DISTRIBUTIONS =====
        self.cd_input_dist = tfd.Bernoulli(probs=self.cd_keep, dtype=tf.bool)
        self.cd_pass_dist = tfd.Bernoulli(probs=self.cd_pass_rate, dtype=tf.bool)
        self.sv_input_dist = tfd.Bernoulli(probs=self.sv_keep, dtype=tf.bool)

        # keep track of when ramping ends
        kl_last_ramp_epoch = cfg.TRAIN.KL.START_EPOCH + cfg.TRAIN.KL.INCREASE_EPOCH
        l2_last_ramp_epoch = cfg.TRAIN.L2.START_EPOCH + cfg.TRAIN.L2.INCREASE_EPOCH
        self.last_ramp_epoch = max([l2_last_ramp_epoch, kl_last_ramp_epoch])

        # attach the config to LFADS
        cfg.freeze()
        self.cfg = cfg

        # wrap the training call with SV and CD
        self._build_wrapped_call()

        # =========================================
        # ===== END OF SETTING UP LFADS MODEL =====
        # =========================================

        # specify the metrics to be logged to CSV (and TensorBoard if chosen)
        self.logging_metrics = [
            'epoch', # epoch of training
            'step', # step of training
            'loss', # total loss on training data
            'nll_heldin', # negative log likelihood on seen training data
            'nll_heldout', # negative log likelihood on unseen training data
            'smth_nll_heldin', # smoothed negative log likelihood on seen training data
            'smth_nll_heldout', # smoothed negative log likelihood on unseen training data
            'wt_kl', # total weighted KL penalty on training data
            'wt_co_kl', # weighted KL penalty on controller output for training data
            'wt_ic_kl', # weighted KL penalty on initial conditions for training data
            'wt_l2', # weighted L2 penalty of the recurrent kernels
            'gnorm', # global norm of the gradient
            'lr', # learning rate
            'kl_wt', # percentage of the weighted KL penalty to applied
            'l2_wt', # percentage of the weighted L2 penalty being applied
            'val_loss', # total loss on validation data
            'val_nll_heldin', # negative log likelihood on seen validation data
            'val_nll_heldout', # negative log likelihood on unseen validation data
            'smth_val_nll_heldin', # smoothed negative log likelihood on seen validation data
            'smth_val_nll_heldout', # smoothed negative log likelihood on unseen validation data
            'val_wt_kl', # total weighted KL penalty on validation data
            'val_wt_co_kl', # weighted KL penalty on controller output for validation data
            'val_wt_ic_kl', # weighted KL penalty on initial conditions for validation data
            'val_wt_l2',
        ]

        # don't log the HPs by default because they are static without PBT
        cfg_dict = flatten(yaml.safe_load(cfg.dump()))
        self.logging_hps = sorted(cfg_dict.keys()) if cfg.TRAIN.LOG_HPS else []

        # create the CSV header if this is a new train_data.csv
        csv_log = logging_conf['handlers']['csv']['filename']
        if not self.from_existing and os.stat(csv_log).st_size == 0:
            self.csv_lgr.info(','.join(self.logging_metrics + self.logging_hps))

        # load the training dataframe
        train_data_path = path.join(self.cfg.TRAIN.MODEL_DIR, 'train_data.csv')
        self.train_df = pd.read_csv(train_data_path, index_col='epoch')

        # create the metrics for saving data during training
        self.all_metrics = {name: tf.keras.metrics.Mean() for name in self.logging_metrics}

        # load the datasets
        self.load_datasets_from_file(cfg.TRAIN.DATA.DIR, cfg.TRAIN.DATA.PREFIX)

        # init training params (here so they are consistent between training sessions)
        self.cur_epoch = tf.Variable(0, trainable=False)
        self.cur_step = tf.Variable(0, trainable=False)
        self.cur_patience = tf.Variable(0, trainable=False)
        self.train_status = '<TRAINING>'
        self.prev_results = {}

        # tracking variables for early stopping
        if self.last_ramp_epoch > self.cur_epoch:
            self.train_status = '<RAMPING>'
        # build the graphs for forward pass and for training
        self.build_graph()
    
    def build_graph(self):
        # ===== AUTOGRAPH FUNCTIONS =====
        # compile the `_step` function into a graph for better speed
        data_shape, _, ext_input_shape, name_shape, beh_shape = self.get_input_shapes()
        # single step of training or validation
        self._graph_step = tf.function(
            func=self._step,
            input_signature=[
                BatchInput(
                    tf.TensorSpec(shape=data_shape), # data used for reconstruction
                    tf.TensorSpec(shape=data_shape, dtype=tf.bool), # mask for sample validation
                    tf.TensorSpec(shape=ext_input_shape), # external inputs
                    tf.TensorSpec(shape=name_shape, dtype=tf.string), # dataset name
                    tf.TensorSpec(shape=beh_shape))]
        )
        # forward pass of LFADS
        self.graph_call = tf.function(
            func=self.call,
            input_signature=[
                LFADSInput(
                    tf.TensorSpec(shape=data_shape),
                    tf.TensorSpec(shape=ext_input_shape),
                    tf.TensorSpec(shape=name_shape, dtype=tf.string),
                    tf.TensorSpec(shape=beh_shape)),
                tf.TensorSpec(shape=[], dtype=tf.bool)])

    def get_config(self):
        """ Get the entire configuration for this LFADS model.

        See the TensorFlow documentation for an explanation of serialization: 
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects

        Returns
        -------
        dict
            A dictionary containing the configuration node.
        """
        return {'cfg_node': self.cfg}

    @classmethod
    def from_config(cls, config):
        """ Initialize an LFADS model from this config.

        See the TensorFlow documentation for an explanation of serialization: 
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        
        Returns
        -------
        lfads_tf2.models.LFADS
            An LFADS model from this config node.
        """
        return cls(cfg_node=config['cfg_node'])

    def update_config(self, config):
        """Updates the configuration of the entire LFADS model.
        
        Updates configuration variables of the model. 
        Primarily used for updating hyperparameters during PBT.

        Parameters
        ----------
        config : dict
            A dictionary containing the new configuration node.

        """
        node = config['cfg_node']
        old_cfg = self.cfg.clone()
        # keep the original model directory
        node.defrost()
        node.TRAIN.MODEL_DIR = old_cfg.TRAIN.MODEL_DIR
        node.freeze()
        # Set the new config
        self.cfg = node
        # Update decoder training appropriately
        encoders_only = node.TRAIN.ENCODERS_ONLY
        self.decoder.trainable = not encoders_only
        if encoders_only ^ old_cfg.TRAIN.ENCODERS_ONLY:
            # If we are swapping ENCODERS_ONLY, reset the optimizer momentum
            # and retrace the graph (learning rate updated below)
            self.lgr.warn(f'`TRAIN.ENCODERS_ONLY` flipped to {encoders_only}. ' \
                'Resetting the optimizer and retracing the graph.')
            for var in self.optimizer.variables():
                var.assign(tf.zeros_like(var))
            self.build_graph()
        self.learning_rate.assign(node.TRAIN.LR.INIT)
        self.dropout_rate.assign(node.MODEL.DROPOUT_RATE)
        self.cd_keep.assign(1 - node.MODEL.CD_RATE)
        self.cd_pass_rate.assign(node.MODEL.CD_PASS_RATE)
        self.sv_keep.assign(1 - node.MODEL.SV_RATE)
        self.max_grad_norm.assign(node.TRAIN.MAX_GRAD_NORM)
        self.loss_scale.assign(node.TRAIN.LOSS_SCALE)
        self.kl_ic_weight.assign(node.TRAIN.KL.IC_WEIGHT)
        self.kl_co_weight.assign(node.TRAIN.KL.CO_WEIGHT)
        self.encoder.update_config(config)
        self.decoder.update_config(config)
        # Reset previous training results
        self.train_df = self.train_df.iloc[0:0]
        self.cur_patience.assign(0)
        self.prev_results = {}
        # Overwrite the configuration file
        # TODO: Add tracking of past configs and record cfg index in train_df
        ms_path = path.join(node.TRAIN.MODEL_DIR, 'model_spec.yaml')
        with open(ms_path, 'w') as ms_file:
            ms_file.write(node.dump())

    def _update_metrics(self, metric_values, batch_size=1):
        """Updates the `self.all_metrics` dictionary with a new batch of values.
        
        Parameters
        ----------
        metric_values : dict
            A dict of metric updates as key-value pairs. Contains only keys 
            found in `self.all_metrics`.
        batch_size : int, optional
            The length of the batch of data used to calculate this metric. 
            Defaults to a weight of 1 for metrics that do not involve the 
            size of the batch.
        """
        for name, value in metric_values.items():
            # weight each metric observation by size of the corresponding batch
            self.all_metrics[name].update_state(value, sample_weight=batch_size)

    def call(self, lfads_input, use_logrates=tf.constant(False), **kwargs):
        """ Performs the forward pass on the LFADS object using one sample 
        from the posteriors. The graph mode version of this function 
        (`self.graph_call`) should be used when speed is preferred.

        NOTE: Overloadable

        Parameters
        ----------
        lfads_input : lfads_tf2.tuples.LFADSInput
            A namedtuple of tensors containing the encoder inputs and 
            external inputs.
        use_logrates : bool, optional
            Whether to return logrates, which are helpful for numerical 
            stability of loss during training, by default False.

        Returns
        -------
        lfads_tf2.tuples.LFADSOutput
            A namedtuple of tensors containing rates and posteriors.
        """

        # separate the inputs
        enc_input, ext_input, dataset_name, behavior = tf.nest.flatten(lfads_input)
        # encode spikes into generator IC distributions and controller inputs
        ic_mean, ic_stddev, ci = self.encoder(enc_input, training=self.training)

        if self.cfg.MODEL.SAMPLE_POSTERIORS:
            # sample from the distribution of initial conditions
            ic_post = tfd.MultivariateNormalDiag(ic_mean, ic_stddev)
            ic = ic_post.sample()
        else:
            # pass mean in deterministic mode
            ic = ic_mean
        # pass initial condition and controller input through decoder network
        dec_input = DecoderInput(
            ic_samp=ic,
            ci=ci,
            ext_input=ext_input)
        dec_output = self.decoder(
            dec_input,
            training=self.training,
            # use_logrates=use_logrates
        )
        co_mean, co_stddev, factors, gen_states, \
            gen_init, gen_inputs, con_states = dec_output

        # compute the rates
        rates = self.transform_factors_to_rates(factors, use_logrates, dataset_name=dataset_name)
        return LFADSOutput(
            rates=rates,
            ic_means=ic_mean,
            ic_stddevs=ic_stddev,
            ic_samps=ic,
            co_means=co_mean,
            co_stddevs=co_stddev,
            factors=factors,
            gen_states=gen_states,
            gen_init=gen_init,
            gen_inputs=gen_inputs,
            con_states=con_states,
            predicted_behavior=behavior)

    def transform_factors_to_rates(self, factors, use_logrates=tf.constant(False), **kwargs):
        logrates = self.rate_linear(factors)
        rates = tf.exp(logrates)

        return logrates if use_logrates else rates

    def load_datasets_from_file(self, data_dir, prefix):
        """A wrapper that loads LFADS datasets from a file.
        
        This function is used for loading datasets into LFADS, 
        including datasets that LFADS was not trained on. LFADS 
        functions like `train` or `sample_and_average` will use 
        whatever data has been loaded by this function. This 
        function is a wrapper around `load_datasets_from_arrays`.
        
        Note
        ----
        This function currently only loads data from one 
        file at a time.

        Parameters
        ----------
        data_dir : str
            The directory containing the data files.
        prefix : str, optional
            The prefix of the data files to be loaded from, 
            by default ''

        See Also
        --------
        lfads_tf2.models.LFADS.load_datasets_from_arrays : 
            Creates tf.data.Dataset objects to use with LFADS.
        """

        self.lgr.info(f"Loading datasets with prefix {prefix} from {data_dir}")
        # load the spiking data from the data file
        data, filenames = load_data(
            data_dir, prefix=prefix, with_filenames=True)
        train_data = {filenames[ii]:data[ii][0] for ii in range(0, len(filenames))}
        valid_data = {filenames[ii]:data[ii][1] for ii in range(0, len(filenames))}

        # save the dataset names as an attribute of the model 
        self.ds_names = filenames 

        # Load any external inputs
        if self.cfg.MODEL.EXT_INPUT_DIM > 0:
            ext_inputs, filenames = load_data(
                data_dir, prefix=prefix, signal='ext_input', with_filenames=True)
            train_ext = {filenames[ii]:ext_inputs[ii][0] for ii in range(0, len(filenames))}
            valid_ext = {filenames[ii]:ext_inputs[ii][1] for ii in range(0, len(filenames))}
        else:
            train_ext, valid_ext = None, None

        # Load any behavioral data 
        if self.cfg.MODEL.BEHAVIOR_DIM > 0:
            behavior, filenames = load_data(
                data_dir, prefix=prefix, signal='behavior', with_filenames=True)
            train_beh = {filenames[ii]:behavior[ii][0] for ii in range(0, len(filenames))}
            valid_beh = {filenames[ii]:behavior[ii][1] for ii in range(0, len(filenames))}
        else:
            train_beh, valid_beh = None, None

        # Load any training and validation indices
        try:
            inds, filenames = load_data(
                data_dir, prefix=prefix, signal='inds', with_filenames=True)
            train_inds = {filenames[ii]:inds[ii][0] for ii in range(0, len(filenames))}
            valid_inds = {filenames[ii]:inds[ii][1] for ii in range(0, len(filenames))}
        except AssertionError:
            train_inds, valid_inds = None, None

        # create the dataset objects
        loadable_data = LoadableData(
            train_data=train_data,
            valid_data=valid_data,
            train_ext_input=train_ext,
            valid_ext_input=valid_ext,
            train_behavior=train_beh,
            valid_behavior=valid_beh,
            train_inds=train_inds,
            valid_inds=valid_inds)
        self.load_datasets_from_arrays(loadable_data)

    def load_datasets_from_arrays(self, loadable_data):
        """Creates TF datasets and attaches them to the LFADS object.

        This function builds dataset objects from input arrays.
        The datasets are used for shuffling, batching, data 
        augmentation, and more. These datasets are used by 
        both posterior sampling and training functions.

        Parameters
        ----------
        loadable_data : lfads_tf2.tuples.LoadableData
            A namedtuple containing the input data and external
            inputs for the training and validation sets. See 
            definition of LoadableData for more detail.

        See Also
        --------
        lfads_tf2.models.LFADS.load_datasets_from_file : 
            A wrapper around this function that loads from a file.

        """

        train_data, valid_data, train_ext, valid_ext, \
            train_behavior, valid_behavior, train_inds, valid_inds = loadable_data

        data_keys = train_data.keys()

        if train_ext is None or valid_ext is None:
            # use empty tensors if there are no inputs
            train_ext = {
                k : tf.zeros(train_data[k].shape[:-1] + (0,)) for k in data_keys}
            valid_ext = {
                k : tf.zeros(valid_data[k].shape[:-1] + (0,)) for k in data_keys}
        
        if train_behavior is None or valid_behavior is None:
            # use empty tensors if there are no inputs
            train_behavior = {
                k : tf.zeros(train_data[k].shape[:-1] + (0,)) for k in data_keys}
            valid_behavior = {
                k : tf.zeros(valid_data[k].shape[:-1] + (0,)) for k in data_keys}

        # create the sample validation masks
        sv_seed = self.cfg.MODEL.SV_SEED
        train_sv_mask = { 
            k : self.sv_input_dist.sample(
                sample_shape=tf.shape(train_data[k]), seed=sv_seed)
            for k in data_keys
        }
        valid_sv_mask = {
            k : self.sv_input_dist.sample(
                sample_shape=tf.shape(valid_data[k]), seed=sv_seed)
            for k in data_keys
        }
        # package up the data into tuples and use to build datasets
        self.train_tuple = {
            k : BatchInput(
                data=train_data[k],
                sv_mask=train_sv_mask[k],
                ext_input=train_ext[k],
                dataset_name=np.array([k]*train_data[k].shape[0]), # all fields must have same first dimension
                behavior=train_behavior[k])
            for k in data_keys
        }
        self.valid_tuple = {
            k : BatchInput(
                data=valid_data[k],
                sv_mask=valid_sv_mask[k],
                ext_input=valid_ext[k],
                dataset_name=np.array([k]*valid_data[k].shape[0]), # all fields must have same first dimension
                behavior=valid_behavior[k])
            for k in data_keys
        }

        # concatenate all of the train tuples together
        self.all_train_tuples = BatchInput(
            data=np.vstack([self.train_tuple[k].data for k in data_keys]),
            sv_mask=np.vstack([self.train_tuple[k].sv_mask for k in data_keys]),
            ext_input=np.vstack([self.train_tuple[k].ext_input for k in data_keys]),
            dataset_name=np.concatenate([self.train_tuple[k].dataset_name for k in data_keys]),
            behavior=np.vstack([self.train_tuple[k].behavior for k in data_keys])
        )

        # concatenate all of the valid tuples together
        self.all_valid_tuples = BatchInput(
            data=np.vstack([self.valid_tuple[k].data for k in data_keys]),
            sv_mask=np.vstack([self.valid_tuple[k].sv_mask for k in data_keys]),
            ext_input=np.vstack([self.valid_tuple[k].ext_input for k in data_keys]),
            dataset_name=np.concatenate([self.valid_tuple[k].dataset_name for k in data_keys]),
            behavior=np.vstack([self.valid_tuple[k].behavior for k in data_keys])
        )

        # create the datasets to batch the data, masks, and input
        self._train_ds = tf.data.Dataset.from_tensor_slices(self.all_train_tuples)
        self._valid_ds = tf.data.Dataset.from_tensor_slices(self.all_valid_tuples)

        # save the indices
        self.train_inds, self.valid_inds = train_inds, valid_inds

    def add_sample_validation(self, model_call):
        """Applies sample validation to a model-calling function.
        
        This decorator applies sample validation to a forward pass 
        through the model. It sets a certain proportion of the input 
        data elements to zero (heldout) and scales up the remaining 
        elements (heldin). It then computes NLL on the heldin 
        samples and the heldout samples separately. `nll_heldin` is 
        used for optimization and `nll_heldout` is used as a metric 
        to detect overfitting to spikes.

        Parameters
        ----------
        model_call : callable
            A callable function with inputs and outputs identical to 
            `LFADS.batch_call`.
        
        Returns
        -------
        callable
            A wrapper around `model_call` that computes heldin and 
            heldout NLL and returns the posterior parameters.
        """

        def sv_step(batch):
            # unpack the batch
            data, sv_mask, *_ = batch
            heldin_mask, heldout_mask = sv_mask, tf.logical_not(sv_mask)
            # set the heldout data to zero and scale up heldin data
            wt_mask = tf.cast(heldin_mask, tf.float32) / self.sv_keep
            heldin_data = data * wt_mask
            # perform the forward pass on the heldin data
            new_batch = BatchInput(
                data=heldin_data,
                sv_mask=batch.sv_mask,
                ext_input=batch.ext_input,
                dataset_name=batch.dataset_name,
                behavior=batch.behavior)
            logrates, posterior_params = model_call(new_batch)
            # compute the nll of the observed samples
            nll_heldin = self.neg_log_likelihood(data, logrates, wt_mask)
            if self.sv_keep < 1:
                # exclude the observed samples from the nll_heldout calculation
                wt_mask = tf.cast(heldout_mask, tf.float32) / (1-self.sv_keep)
                nll_heldout = self.neg_log_likelihood(data, logrates, wt_mask)
            else:
                nll_heldout = np.nan

            return nll_heldin, nll_heldout, posterior_params

        return sv_step


    def add_coordinated_dropout(self, model_call):
        """Applies coordinated dropout to a model-calling function.
        
        A decorator that applies coordinated dropout to a forward pass 
        through the model. It sets a certain proportion of the input 
        data elements to zero and scales up the remaining elements. When 
        the model is being trained, it can only backpropagate gradients 
        for matrix elements it didn't see at the input. The function 
        outputs a gradient mask that is incorporated by the sample 
        validation wrapper.

        Parameters
        ----------
        model_call : callable
            A callable function with inputs and outputs identical to 
            `LFADS.batch_call`.
        
        Returns
        -------
        callable
            A wrapper around `model_call` that blocks matrix elements 
            before the call and passes a mask to block gradients of the 
            observed matrix elements.

        """

        def block_gradients(input_data, keep_mask):
            keep_mask = tf.cast(keep_mask, tf.float32)
            block_mask = 1 - keep_mask
            return tf.stop_gradient(input_data * block_mask) + input_data * keep_mask

        def cd_step(batch):
            input_data = batch.data
            # samples a new coordinated dropout mask at every training step
            cd_mask = self.cd_input_dist.sample(sample_shape=tf.shape(input_data))
            pass_mask = self.cd_pass_dist.sample(sample_shape=tf.shape(input_data))
            grad_mask = tf.logical_or(tf.logical_not(cd_mask), pass_mask)
            # mask and scale up the post-CD input so it has the same sum as the original data
            cd_masked_data = input_data * tf.cast(cd_mask, tf.float32)
            cd_masked_data /= self.cd_keep
            # perform a forward pass on the cd masked data
            new_batch = BatchInput(
                data=cd_masked_data,
                sv_mask=batch.sv_mask,
                ext_input=batch.ext_input,
                dataset_name=batch.dataset_name,
                behavior=batch.behavior)
            logrates, posterior_params = model_call(new_batch)
            # block the gradients with respect to the masked outputs
            logrates = block_gradients(logrates, grad_mask)
            return logrates, posterior_params

        return cd_step

    def batch_to_LFADSInput(self, batch):
        """Converts a BatchInput named tuple to an LFADSInput named tuple.
        Useful to override for input preprocessing (e.g. dimensionality
        reduction).

        NOTE: Overloadable

        Parameters
        ----------
        batch : lfads_tf2.tuples.BatchInput
            A namedtuple contining tf.Tensors for spiking data, 
            external inputs, and a sample validation mask.
        Returns
        -------
        lfads_tf2.tuples.LFADSInput
            A namedtuple contining tf.Tensors for encoder input and
            external input
        """
        data, _, ext_input, dataset_name, behavior = batch
        # Create the low-D input for LFADS
        lfads_input = LFADSInput(
            enc_input=data, ext_input=ext_input, dataset_name=dataset_name, behavior=behavior)
        return lfads_input

    def batch_call(self, batch):
        """Performs the forward pass on a batch of input data.

        This is a wrapper around the forward pass of LFADS, meant to be 
        more compatible with the coordinated dropout and sample 
        validation wrappers.

        Parameters
        ----------
        batch : lfads_tf2.tuples.BatchInput
            A namedtuple contining tf.Tensors for spiking data, 
            external inputs, and a sample validation mask.

        Returns
        -------
        tf.Tensor
            A BxTxN tensor of log-rates, where B is the batch size, 
            T is the number of time steps, and N is the number of neurons.
        tuple of tf.Tensor
            Four tensors corresponding to the posteriors - `ic_mean`, 
            `ic_stddev`, `co_mean`, `co_stddev`.
        tf.Tensor
            A BxTxN boolean tensor that indicates whether matrix elements 
            should be used to calculate gradients.

        """
        lfads_input = self.batch_to_LFADSInput(batch)
        if self.cfg.TRAIN.EAGER_MODE:
            output = self.call(lfads_input, use_logrates=True)
        else:
            output = self.graph_call(lfads_input, use_logrates=True)

        posterior_params = (
            output.ic_means,
            output.ic_stddevs,
            output.co_means,
            output.co_stddevs)

        return output.rates, posterior_params


    def neg_log_likelihood(self, data, logrates, wt_mask=None):
        """Computes the log likelihood of the data, given 
        predicted rates. 

        This function computes the average negative log likelihood 
        of the spikes in this batch, given the rates that LFADS 
        predicts for the samples.

        Parameters
        ----------
        data : tf.Tensor
            A BxTxN tensor of spiking data.
        logrates : tf.Tensor
            A BxTxN tensor of log-rates.
        wt_mask : tf.Tensor
            A weighted mask to apply to the likelihoods.

        Returns
        -------
        tf.Tensor
            A scalar tensor representing the mean negative 
            log-likelihood of these spikes.        
        
        """
        if wt_mask is None:
            wt_mask = tf.ones_like(data)
        nll_all = tf.nn.log_poisson_loss(data, logrates, compute_full_loss=True)
        nll_masked = nll_all * wt_mask
        if self.cfg.TRAIN.NLL_MEAN:
            # Average over all elements of the data tensor
            nll = tf.reduce_mean(nll_masked)
        else:
            # Sum over inner dimensions, average over batch dimension
            nll = tf.reduce_mean(tf.reduce_sum(nll_masked, axis=[1,2]))
        return nll


    def weighted_kl_loss(self, ic_mean, ic_stddev, co_mean, co_stddev):
        """Computes the KL loss based on the priors.
        
        This function computes the weighted KL loss of all of 
        the posteriors. The KL of the initial conditions is computed 
        directly, but the KL of the controller output distributions
        is approximated via sampling.

        Parameters
        ----------
        ic_mean : tf.Tensor
            A BxIC_DIM tensor of initial condition means.
        ic_stddev : tf.Tensor
            A BxIC_DIM tensor of initial condition standard deviations.
        co_mean : tf.Tensor
            A BxTxCO_DIM tensor of controller output means.
        co_stddev : tf.Tensor
            A BxTxCO_DIM tensor of controller output standard deviations.
        
        Returns
        -------
        tf.Tensor
            A scalar tensor of the total KL loss of the model.

        """
        ic_post, co_post = self.make_posteriors(
            ic_mean, ic_stddev, co_mean, co_stddev)
        # Create the IC priors
        ic_prior_stddev = tf.exp(0.5 * self.ic_prior_logvar)
        ic_prior = tfd.MultivariateNormalDiag(self.ic_prior_mean, ic_prior_stddev)
        # compute KL for the IC's analytically
        ic_kl_batch = tfd.kl_divergence(ic_post, ic_prior)
        wt_ic_kl = tf.reduce_mean(ic_kl_batch) * self.kl_ic_weight

        # compute KL for the CO's via sampling
        wt_co_kl = 0.0
        if self.use_con:
            sample = co_post.sample()
            log_q = co_post.log_prob(sample)
            log_p = self.co_prior.log_prob(sample)
            wt_co_kl = tf.reduce_mean(log_q - log_p) * self.kl_co_weight

        batch_size = ic_mean.shape[0]
        if self.training:
            self._update_metrics({
                'wt_ic_kl': wt_ic_kl,
                'wt_co_kl': wt_co_kl}, batch_size)
        else:
            self._update_metrics({
                'val_wt_ic_kl': wt_ic_kl,
                'val_wt_co_kl': wt_co_kl}, batch_size)

        return wt_ic_kl + wt_co_kl

    def weighted_l2_loss(self):
        """ Computes the L2 loss for the LFADS model.
        
        This function computes the weighted L2 loss across all of the 
        recurrent kernels in LFADS. It is implemented outside of the 
        training function for extensibility.

        NOTE: Overloadable

        Returns
        -------
        tf.Tensor
            A scalar tensor of the total L2 loss of the model.

        """
        rnn_losses = self.encoder.losses + self.decoder.losses
        l2 = tf.reduce_sum(rnn_losses) / \
            (self.model_recurrent_size + tf.keras.backend.epsilon())
        return l2

    def _step(self, batch):
        """ Performs a step of training or validation.
        
        Depending on the state of the boolean `self.training` variable, 
        this function will either perform a step of training or a step
        of validation. This entails a forward pass through the model on 
        a batch of data, calculating and logging losses, and possibly 
        taking a training step.
        
        Parameters
        ----------
        batch : lfads_tf2.tuples.BatchInput
            A namedtuple contining tf.Tensors for spiking data, 
            external inputs, and a sample validation mask.
        
        """

        # ----- TRAINING STEP -----
        if self.training:
            with tf.GradientTape() as tape:
                # perform the forward pass, using SV and / or CD as necessary
                nll_heldin, nll_heldout, posterior_params = self.train_call(batch)

                l2 = self.weighted_l2_loss()
                kl = self.weighted_kl_loss(*posterior_params)

                loss = nll_heldin + self.l2_ramping_weight * l2 \
                    + self.kl_ramping_weight * kl

                scaled_loss = self.loss_scale * loss

            # compute gradients and descend
            gradients = tape.gradient(scaled_loss, self.trainable_variables)
            gradients, gnorm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # record training statistics
            self._update_metrics({
                'loss': loss,
                'nll_heldin': nll_heldin,
                'nll_heldout': nll_heldout,
                'wt_kl': kl,
                'wt_l2': l2,
                'gnorm': gnorm,
            }, batch_size=batch.data.shape[0])

        # ----- VALIDATION STEP -----
        else:
            # perform the forward pass through the network
            nll_heldin, nll_heldout, posterior_params = self.val_call(batch)

            l2 = self.weighted_l2_loss()
            kl = self.weighted_kl_loss(*posterior_params)

            loss = nll_heldin + self.l2_ramping_weight * l2 \
                + self.kl_ramping_weight * kl

            # record training statistics
            self._update_metrics({
                'val_loss': loss,
                'val_nll_heldin': nll_heldin,
                'val_nll_heldout': nll_heldout,
                'val_wt_kl': kl,
                'val_wt_l2': l2,
            }, batch_size=batch.data.shape[0])

    def make_posteriors(self, ic_mean, ic_stddev, co_mean, co_stddev):
        """ Creates posterior distributions from their parameters.
        
        Parameters
        ----------
        ic_mean : tf.Tensor
            A BxIC_DIM tensor of initial condition means.
        ic_stddev : tf.Tensor
            A BxIC_DIM tensor of initial condition standard deviations.
        co_mean : tf.Tensor
            A BxTxCO_DIM tensor of controller output means.
        co_stddev : tf.Tensor
            A BxTxCO_DIM tensor of controller output standard deviations.

        Returns
        -------
        tfd.MultivariateNormalDiag
            The initial condition posterior distribution.
        tfd.Independent(tfd.MultivariateNormalDiag)
            The controller output posterior distribution.

        """
        ic_post = tfd.MultivariateNormalDiag(ic_mean, ic_stddev)
        co_post = tfd.Independent(tfd.MultivariateNormalDiag(co_mean, co_stddev))
        return ic_post, co_post


    def _build_wrapped_call(self):
        """Assembles the forward pass using SV and CD wrappers.

        Conveniently wraps the forward pass of LFADS with coordinated
        dropout and sample validation to allow automatic application 
        of these paradigms.
        """
        train_call = self.batch_call
        if self.cd_keep < 1:
            train_call = self.add_coordinated_dropout(train_call)
        train_call = self.add_sample_validation(train_call)
        val_call = self.add_sample_validation(self.batch_call)

        if self.cfg.TRAIN.EAGER_MODE:
            self.train_call = train_call
            self.val_call = val_call
        else:
            data_shape, _, ext_input_shape, name_shape, beh_shape = self.get_input_shapes()
            # single step of training or validation
            input_signature=[
                BatchInput(
                    tf.TensorSpec(shape=data_shape), # data used for reconstruction
                    tf.TensorSpec(shape=data_shape, dtype=tf.bool), # mask for sample validation
                    tf.TensorSpec(shape=ext_input_shape), # external inputs
                    tf.TensorSpec(shape=name_shape, dtype=tf.string),
                    tf.TensorSpec(shape=beh_shape))] # dataset name 
            self.train_call = tf.function(func=train_call, input_signature=input_signature)
            self.val_call = tf.function(func=val_call, input_signature=input_signature)


    def update_learning_rate(self):
        """Updates the learning rate of the optimizer. 
        
        Calculates the learning rate, based on model improvement
        over the last several epochs. After the last ramping epoch, we 
        start checking whether the current loss is worse than the worst
        in the previous few epochs. The exact number of epochs to 
        compare is determined by PATIENCE. If the current epoch is worse, 
        then we decline the learning rate by multiplying the decay factor. 
        This can only happen at most every PATIENCE epochs.

        Returns
        -------
        float
            The new learning rate.
        """

        cfg = self.cfg.TRAIN.LR
        prev_epoch = self.train_df.index[-1] if len(self.train_df) > 0 else 0
        if 0 < cfg.DECAY < 1 and prev_epoch > self.last_ramp_epoch + cfg.PATIENCE:
            # allow the learning rate to decay at a max of every cfg.PATIENCE epochs
            epochs_since_decay = (self.train_df.lr == self.train_df.at[prev_epoch, 'lr']).sum()
            if epochs_since_decay >= cfg.PATIENCE:
                # compare the current val_loss to the max over a window of previous epochs
                winmax_val_loss = self.train_df.iloc[-(cfg.PATIENCE+1):-1].val_loss.max()
                cur_val_loss = self.train_df.at[prev_epoch, 'val_loss']
                # if the current val_loss is greater than the max in the window, decay LR
                if cur_val_loss > winmax_val_loss:
                    new_lr = max([cfg.DECAY * self.learning_rate.numpy(), cfg.STOP])
                    self.learning_rate.assign(new_lr)
        # report the current learning rate to the metrics
        new_lr = self.learning_rate.numpy()
        self._update_metrics({'lr': new_lr})
        return new_lr


    def update_ramping_weights(self):
        """Updates the ramping weight variables. 
        
        In order to allow the model to learn more quickly, we introduce 
        regularization slowly by linearly increasing the ramping weight 
        as the model is training, to maximum ramping weights of 1. This 
        function computes the desired weight and assigns it to the ramping 
        variables, which are later used in the KL and L2 calculations.
        Ramping is determined by START_EPOCH, or the epoch to start 
        ramping, and INCREASE_EPOCH, or the number of epochs over which 
        to increase to full regularization strength. Ramping can occur 
        separately for KL and L2 losses, based on their respective 
        hyperparameters.

        """

        cfg = self.cfg.TRAIN
        cur_epoch = self.cur_epoch.numpy()
        def compute_weight(start_epoch, increase_epoch):
            if increase_epoch > 0:
                ratio = (cur_epoch - start_epoch) / (increase_epoch + 1)
                clipped_ratio = tf.clip_by_value(ratio, 0, 1)
                return tf.cast(clipped_ratio, tf.float32)
            else:
                return 1.0 if cur_epoch >= start_epoch else 0.0
        # compute the new weights
        kl_weight = compute_weight(cfg.KL.START_EPOCH, cfg.KL.INCREASE_EPOCH)
        l2_weight = compute_weight(cfg.L2.START_EPOCH, cfg.L2.INCREASE_EPOCH)
        # assign values to the ramping weight variables
        self.kl_ramping_weight.assign(kl_weight)
        self.l2_ramping_weight.assign(l2_weight)
        # report the current KL and L2 weights to the metrics
        self._update_metrics({'kl_wt': kl_weight, 'l2_wt': l2_weight})


    def train_epoch(self, loadable_data=None):
        """Trains LFADS for a single epoch.
        
        This function is designed to implement a single unit 
        of training, such that it may be called arbitrarily 
        by the user. It computes the desired loss weights, iterates 
        through the training dataset and validation dataset once, 
        computes and logs losses, checks stopping criteria, saves 
        checkpoints, and reports results in a dictionary. The 
        results of training this epoch are returned as a dictionary.

        Parameters
        ----------
        loadable_data : lfads_tf2.tuples.LoadableData
            A namedtuple containing the input data and external
            inputs for the training and validation sets. See 
            definition of LoadableData for more detail.

        Returns
        -------
        dict
            The results of all metrics evaluated during training.
 
        """
        if loadable_data is not None:
            self.load_datasets_from_arrays(loadable_data)

        cfg = self.cfg.TRAIN
        self.cur_epoch.assign_add(1)
        cur_epoch = self.cur_epoch.numpy()
        cur_lr = self.update_learning_rate()
        self.update_ramping_weights()
        epoch_header_template = ', '.join([
            'Current epoch: {cur_epoch}/{max_epochs}',
            'Steps completed: {cur_step}',
            'Patience: {patience}',
            'Status: {train_status}',
            'LR: {cur_lr:.2E}',
        ])
        epoch_header = epoch_header_template.format(**{
            'cur_epoch': cur_epoch, 
            'cur_step': self.cur_step.numpy(),
            'max_epochs': cfg.MAX_EPOCHS - 1, 
            'patience': self.cur_patience.numpy(), 
            'train_status': self.train_status, 
            'cur_lr': cur_lr,
        })
        self.lgr.info(epoch_header)
        # only use the remainder when it is the full batch
        if cfg.BATCH_SIZE > len(self.all_train_tuples.data):
            samples_per_epoch = len(self.all_train_tuples.data)
            drop_remainder = False
        else:
            samples_per_epoch = (len(self.all_train_tuples.data) // cfg.BATCH_SIZE) * cfg.BATCH_SIZE
            drop_remainder = True
        # only show progress bar when not in `TUNE_MODE`
        if not cfg.TUNE_MODE:
            pbar = Progbar(samples_per_epoch, width=50, unit_name='sample')

        # applying data augmentation
        # use autotune to parallelize computations automatically
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        jitter = self.cfg.TRAIN.DATA.AUGMENT.JITTER_WIDTH
        augment = lambda batch : shuffle_spikes_in_time(batch, jitter)
        train_dataset = (
                    self._train_ds
                    # apply jitter to the training data
                    .map(augment, num_parallel_calls=AUTOTUNE)
                    # shuffle samples with buffer size > # of samples
                    .shuffle(10000)
                    # divide into batches
                    .batch(cfg.BATCH_SIZE, drop_remainder=drop_remainder)
                )
        valid_dataset = (
                    self._valid_ds
                    # apply jitter to valid data
                    .map(augment, num_parallel_calls=AUTOTUNE)
                    # divide into batches
                    .batch(cfg.BATCH_SIZE)
                )
        # main training loop over the batches
        # iterate through training batches
        for batch in train_dataset:
            # set model to training mode
            self.training.assign(True)
            # perform training step
            if cfg.EAGER_MODE:
                self._step(batch)
            else:
                self._graph_step(batch)
            self.cur_step.assign_add(1)
            # update progress bar
            if not cfg.TUNE_MODE:
                pbar.add(len(batch[0]))
        # iterate through validation batches
        for val_batch in valid_dataset:
            # set model to validation mode
            self.training.assign(False)
            # perform validation step
            if cfg.EAGER_MODE:
                self._step(val_batch)
            else:
                self._graph_step(val_batch)

        def smth_metric(name, coef=0.7):
            # calculate exponentially smoothed value for a given metric
            cur_metric = self.all_metrics[name].result().numpy()
            prev_metric = self.prev_results.get('smth_' + name, cur_metric)
            smth_metric = (1 - coef) * prev_metric + coef * cur_metric
            return smth_metric

        # report smoothed values and epoch number to the metrics
        self._update_metrics({
            'smth_nll_heldin': smth_metric('nll_heldin'),
            'smth_nll_heldout': smth_metric('nll_heldout'),
            'smth_val_nll_heldin': smth_metric('val_nll_heldin'),
            'smth_val_nll_heldout': smth_metric('val_nll_heldout'),
        })

        # get the reset the metrics after each epoch (record before breaking the loop)
        results = {name: m.result().numpy() for name, m in self.all_metrics.items()}
        _ = [m.reset_states() for name, m in self.all_metrics.items()]
        results.update({'epoch': cur_epoch, 'step': self.cur_step.numpy()})

        # print the status of the metrics
        train_template = ' - '.join([
            "    loss: {loss:.3f}",
            "    nll_heldin: {nll_heldin:.3f}",
            "    nll_heldout: {nll_heldout:.3f}",
            "    wt_kl: {wt_kl:.2E}",
            "    wt_l2: {wt_l2:.2E}",
            "gnorm: {gnorm:.2E}",
        ])
        val_template = ' - '.join([
            "val_loss: {val_loss:.3f}",
            "val_nll_heldin: {val_nll_heldin:.3f}",
            "val_nll_heldout: {val_nll_heldout:.3f}",
            "val_wt_kl: {val_wt_kl:.2E}",
            "val_wt_l2: {val_wt_l2:.2E}",
        ])
        self.lgr.info(train_template.format(**results))
        self.lgr.info(val_template.format(**results))

        if self.cfg.TRAIN.LOG_HPS:
            # report the HPs for this stretch of training
            cfg_dict = flatten(yaml.safe_load(self.cfg.dump()))
            results.update(cfg_dict)

        # write the metrics and HPs to the in-memory `train_df` for evaluation
        new_results_df = pd.DataFrame({key: [val] for key, val in results.items()}).set_index('epoch')
        self.train_df = pd.concat([self.train_df, copy.deepcopy(new_results_df)])

        # ---------- Check all criteria that could stop the training loop ----------
        def check_max_epochs(train_df):
            pass_check = True
            if cur_epoch >= self.cfg.TRAIN.MAX_EPOCHS:
                pass_check = False
            return pass_check

        def check_nans(train_df):
            """ Check if training should stop because of NaN's """
            loss = train_df.at[cur_epoch, 'loss']
            val_loss = train_df.at[cur_epoch, 'val_loss']
            pass_check = True
            nan_found = np.isnan(loss) or np.isnan(val_loss)
            # Only stop for NaN loss when not in PBT_MODE
            if not cfg.PBT_MODE and nan_found:
                self.lgr.info("STOPPING: NaN found in loss.")
                pass_check = False
            return pass_check

        def check_lr(train_df):
            """ Check if training should stop because of the learning rate. """
            cur_lr = train_df.at[cur_epoch, 'lr']
            pass_check = True
            if cur_lr <= cfg.LR.STOP:
                self.lgr.info(f"STOPPING: Learning rate has reached {cfg.LR.STOP}.")
                pass_check = False
            return pass_check

        def check_earlystop(train_df):
            """ Check if training should stop because the validation loss has not improved. """
            best_epoch = -1
            pass_check = True
            if cur_epoch > self.last_ramp_epoch:
                # find the best epoch only after ramping is complete
                postramp_df = train_df[(train_df['kl_wt'] == 1) & (train_df['l2_wt'] == 1)]
                best_epoch = postramp_df.smth_val_nll_heldin.idxmin()
                # save a checkpoint if this model is the best and beyond `self.last_ramp_epoch`
                # use the `self.train_status` to report the status of early stopping
                if best_epoch == cur_epoch:
                    self.train_status = "<TRAINING>"
                    # save a checkpoint if this model is the best
                    self.lve_manager.save()
                else:
                    self.train_status = "<WAITING>"
            # stop training if `smth_val_nll` does not improve  after `cfg.PATIENCE` epochs
            bounds =  [best_epoch, self.last_ramp_epoch]
            self.cur_patience.assign(max([cur_epoch - max(bounds), 0]))
            if self.cur_patience.numpy() >= cfg.PATIENCE:
                self.lgr.info(f"STOPPING: No improvement in `smth_val_nll_heldin` for {cfg.PATIENCE} epochs.")
                pass_check = False
            return pass_check

        # Run all of the checks for stopping criterion
        check_funcs = [check_max_epochs, check_nans, check_lr, check_earlystop]
        checks_passed = [check_func(self.train_df) for check_func in check_funcs]
        # end the training loop if not all checks passed
        if not all(checks_passed):
            # Indicate that training is over
            results['done'] = True
            # Remove the last (ending) epoch from the `train_df`, in case we restart training
            self.train_df = self.train_df[:-1]
        else:
            # If all checks pass, save a checkpoint
            self.mrc_manager.save()
            # Save the results from the previous successful epoch for comparison
            self.prev_results = copy.deepcopy(results)
            # Add quotation marks so commas inside strings are ignored by csv parser
            convert_for_csv = lambda data: f'\"{data}\"' if type(data) == str else str(data)
            csv_output = [convert_for_csv(results[log_metric]) \
                for log_metric in self.logging_metrics + self.logging_hps]
            # Save the results of successful epochs to the CSV
            self.csv_lgr.info(','.join(csv_output))
        # log the metrics for tensorboard
        if self.cfg.TRAIN.USE_TB:

            def pca(data):
                # reduces the data to its first principal component for visualization
                pca_data = tf.reshape(data, (data.shape[0], -1)) if tf.rank(data) > 2 else data
                pca_obj = PCA(n_components=1)
                return pca_obj.fit(pca_data).transform(pca_data)

            def make_figure(data, rates, co_post):
                figure = plt.figure(figsize=(10,10))
                if co_post.event_shape[-1] > 0:
                    plt.subplot(2, 2, 1, title='Single Neuron')
                    plt.plot(data[0,:,0], label='spikes')
                    plt.plot(rates[0,:,0], label='rates')
                    plt.subplot(2, 2, 2, title='Sample Controller Output')
                    plt.plot(co_post.sample()[0,:,0])
                else:
                    plt.subplot(2, 1, 1, title='Single Neuron')
                    plt.plot(data[0,:,0], label='spikes')
                    plt.plot(rates[0,:,0], label='rates')
                plt.subplot(2, 2, 3, title='All Spikes')
                plt.imshow(tf.transpose(data[0,:,:]))
                plt.subplot(2, 2, 4, title='All Rates')
                plt.imshow(tf.transpose(rates[0,:,:]))
                return figure

            def figure_to_tf(figure):
                # save the plot to a PNG in memory
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(figure)
                # convert the png buffer to TF image
                buf.seek(0)
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                return tf.expand_dims(image, 0)

            with self.summary_writer.as_default():
                tf.summary.experimental.set_step(cur_epoch)
                with tf.name_scope('lfads'):
                    for name, value in results.items():
                        # record all of the results in tensorboard
                        if type(value) == float:
                            tf.summary.scalar(name, value)
                    if cur_epoch % 5 == 0:
                        # prepare the data for the forward pass - NOTE: no sample validation is used here.
                        train_input = self.batch_to_LFADSInput(batch)
                        val_input = self.batch_to_LFADSInput(val_batch)
                        # do a forward pass on all data in eager mode so we can log distributions to tensorboard
                        t_output = self.call(train_input)
                        t_rates = t_output.rates
                        posterior_params = (
                            t_output.ic_means,
                            t_output.ic_stddevs,
                            t_output.co_means,
                            t_output.co_stddevs)
                        t_ic_post, t_co_post = self.make_posteriors(*posterior_params)
                        v_output = self.call(val_input)
                        v_rates = v_output.rates
                        posterior_params = (
                            v_output.ic_means,
                            v_output.ic_stddevs,
                            v_output.co_means,
                            v_output.co_stddevs)
                        v_ic_post, v_co_post = self.make_posteriors(*posterior_params)
                        # make figures every fifth epoch - uses a different backend for matplotlib
                        plt.switch_backend('Agg')
                        train_fig = make_figure(self.all_train_tuples.data, t_rates, t_co_post)
                        tf.summary.image('train_ouput', figure_to_tf(train_fig))
                        val_fig = make_figure(self.all_valid_tuples.data, v_rates, v_co_post)
                        tf.summary.image('val_output', figure_to_tf(val_fig))
                        plt.switch_backend('agg')
                        # compute histograms of controller outputs
                        with tf.name_scope('prior'):
                            tf.summary.histogram('ic_mean', self.ic_prior_mean)
                            tf.summary.histogram('ic_stddev', tf.exp(0.5 * self.ic_prior_logvar))
                            if self.use_con:
                                tf.summary.histogram('priors/co_logtau', self.logtaus)
                                tf.summary.histogram('priors/co_lognvar', self.lognvars)
                        with tf.name_scope('post'):
                            with tf.name_scope('train'):
                                tf.summary.histogram('ic_mean', t_ic_post.mean())
                                tf.summary.histogram('ic_stddev', t_ic_post.stddev())
                                tf.summary.histogram('ic_mean_PC1', pca(t_ic_post.mean()))
                                tf.summary.histogram('ic_stddev_PC1', pca(t_ic_post.stddev()))
                                if self.use_con:
                                    tf.summary.histogram('co_mean', t_co_post.mean())
                                    tf.summary.histogram('co_stddev', t_co_post.stddev())
                                    tf.summary.histogram('co_mean_PC1', pca(t_co_post.mean()))
                                    tf.summary.histogram('co_stddev_PC1', pca(t_co_post.stddev()))
                            with tf.name_scope('valid'):
                                tf.summary.histogram('ic_mean', v_ic_post.mean())
                                tf.summary.histogram('ic_stddev', v_ic_post.stddev())
                                tf.summary.histogram('ic_mean_PC1', pca(v_ic_post.mean()))
                                tf.summary.histogram('ic_stddev_PC1', pca(v_ic_post.stddev()))
                                if self.use_con:
                                    tf.summary.histogram('co_mean', v_co_post.mean())
                                    tf.summary.histogram('co_stddev', v_co_post.stddev())
                                    tf.summary.histogram('co_mean_PC1', pca(v_co_post.mean()))
                                    tf.summary.histogram('co_stddev_PC1', pca(v_co_post.stddev()))

                self.summary_writer.flush()

        # remove NaN's, which can cause bugs for TensorBoard
        results = {key: val for key, val in results.items() if val != np.nan}
        return results


    def train(self, loadable_data=None):
        """Trains LFADS until any stopping criterion is reached. 
        
        Runs `LFADS.train_epoch` until it reports that training is 
        complete by including `results['done'] == True` in the results 
        dictionary.

        Parameters
        ----------
        loadable_data : lfads_tf2.tuples.LoadableData, optional
            A namedtuple containing the input data and external
            inputs for the training and validation sets. See 
            definition of LoadableData for more detail. By default,
            None uses data that has already been loaded.
        
        Returns
        -------
        dict
            The results of all metrics from the last epoch.
        """

        if loadable_data is not None:
            self.load_datasets_from_arrays(loadable_data)

        # train epochs until the `done` signal is recieved
        self.lgr.info(" Train on {}, validate on {} samples".format(
            len(self.all_train_tuples.data), len(self.all_valid_tuples.data)))
        done = False
        while not done:
            results = self.train_epoch()
            done = results.get('done', False)

        # restore the best model if not using the most recent checkpoints
        if not self.cfg.TRAIN.PBT_MODE and self.lve_manager.latest_checkpoint is not None:
            self.restore_weights()

        # record that the model is trained
        self.is_trained = True

        return results

    def posterior_sample_and_average_call(self, lfads_input, n_samples):
        """ Performs the posterior estimation for the LFADS graph using 
        the input data.
        
        NOTE: Overloadable

        Parameters
        ----------
        lfads_input : lfads_tf2.tuples.LFADSInput
            A namedtuple of tensors containing the data, external inputs, 
            and encoder inputs.
        n_samples : int
            The number of samples to take from the posterior 
            distribution for each datapoint, by default 50.

        Returns
        -------
        list of np.ndarray
            Things that are averaged across samples (most things)
        list of np.ndarray
            IC means and stddevs
        """
        # Unpack the input data
        enc_input, ext_input, dataset_names, behavior = lfads_input
        # for each chop in the dataset, compute the initial conditions distribution
        ic_mean, ic_stddev, ci = self.encoder.graph_call(enc_input)
        ic_post = tfd.MultivariateNormalDiag(ic_mean, ic_stddev)

        # define merging and splitting utilities
        def merge_samp_and_batch(data, batch_dim):
            """ Combines the sample and batch dimensions """
            return tf.reshape(
                data, [n_samples * batch_dim] + tf.unstack(tf.shape(data)[2:]))

        def split_samp_and_batch(data, batch_dim):
            """ Splits up the sample and batch dimensions """
            return tf.reshape(
                data, [n_samples, batch_dim] + tf.unstack(tf.shape(data)[1:]))

        # sample from the posterior and merge sample and batch dimensions
        if self.cfg.MODEL.SAMPLE_POSTERIORS:
            ic_post_samples = ic_post.sample(n_samples)
        else:
            samples_shape = ic_post.sample(n_samples).shape
            ic_post_samples = ic_mean * np.ones(samples_shape) 

        ic_post_samples_merged = merge_samp_and_batch(
            ic_post_samples, len(enc_input))

        # tile and merge the controller inputs and the external inputs
        ci_tiled = tf.tile(tf.expand_dims(ci, axis=0), [n_samples, 1, 1, 1])
        ci_merged = merge_samp_and_batch(ci_tiled, len(enc_input))
        ext_tiled = tf.tile(tf.expand_dims(ext_input, axis=0), [n_samples, 1, 1, 1])
        ext_merged = merge_samp_and_batch(ext_tiled, len(enc_input))

        # pass all samples into the decoder
        dec_input = DecoderInput(
            ic_samp=ic_post_samples_merged,
            ci=ci_merged,
            ext_input=ext_merged)
        output_samples_merged = self.decoder(dec_input)

        # average the outputs across samples
        output_samples = [split_samp_and_batch(t, len(enc_input)) \
            for t in output_samples_merged]
        output = [np.mean(t, axis=0) for t in output_samples]

        # aggregate for each batch
        non_averaged_outputs = [
            ic_mean.numpy(),
            tf.math.log(ic_stddev**2).numpy(),
        ]
        output.append(behavior) # return behavior unchanged when not predicting 
        return output, non_averaged_outputs

    def sample_and_average(self,
                           loadable_data=None,
                           n_samples=50,
                           batch_size=64,
                           ps_filename='posterior_samples.h5',
                           save=True,
                           merge_tv=False):
        """Saves rate estimates to the 'model_dir'.
        
        Performs a forward pass of LFADS, but passes multiple 
        samples from the posteriors, which can be used to get a 
        more accurate estimate of the rates. Saves all output 
        to posterior_samples.h5 in the `model_dir`.

        Parameters
        ----------
        loadable_data : lfads_tf2.tuples.LoadableData, optional
            A namedtuple containing the input data and external
            inputs for the training and validation sets. See 
            definition of LoadableData for more detail. By default,
            None uses data that has already been loaded.
        n_samples : int, optional
            The number of samples to take from the posterior 
            distribution for each datapoint, by default 50.
        batch_size : int, optional
            The number of samples per batch, by default 128.
        ps_filename : str, optional
            The name of the posterior sample file, by default
            'posterior_samples.h5'. Ignored if `save` is False.
        save : bool, optional
            Whether or not to save the posterior sampling output
            to a file, if False will return a tuple of 
            SamplingOutput. By default, True.
        merge_tv : bool, optional
            Whether to merge training and validation output, 
            by default False. Ignored if `save` is True.

        Returns
        -------
        SamplingOutput
            If save is True, return nothing. If save is False, 
            and merge_tv is false, retun SamplingOutput objects 
            training and validation data. If save is False and 
            merge_tv is True, return a single SamplingOutput 
            object.

        """

        if loadable_data is not None:
            self.load_datasets_from_arrays(loadable_data)

        # get the filename for posterior sampling output
        output_file = path.join(
            self.cfg.TRAIN.MODEL_DIR, ps_filename)

        try:
            # remove any pre-existing posterior sampling file
            os.remove(output_file)
            self.lgr.info(
                f"Removing existing posterior sampling file at {output_file}")
        except OSError:
            pass

        if not self.is_trained:
            self.lgr.warn(
                "Performing posterior sampling on an untrained model.")

        # ========== POSTERIOR SAMPLING ==========
        # perform sampling on both training and validation data
        for prefix, dataset in zip(['train_', 'valid_'], [self._train_ds, self._valid_ds]):
            data_len = len(self.all_train_tuples.data) if prefix == 'train_' else len(self.all_valid_tuples.data)

            # initialize lists to store rates
            all_outputs = []
            self.lgr.info("Posterior sample and average on {} segments.".format(data_len))
            if not self.cfg.TRAIN.TUNE_MODE:
                pbar = Progbar(data_len, width=50, unit_name='dataset')
            dataset_names = []
            for batch in dataset.batch(batch_size):
                # convert the batch into LFADS input
                lfads_input = self.batch_to_LFADSInput(batch)
                dataset_names.append(lfads_input.dataset_name.numpy())

                # run the (possibly overloaded) posterior_sample_and_average_call function
                output, non_averaged_outputs = \
                    self.posterior_sample_and_average_call(lfads_input, n_samples)

                all_outputs.append(output + non_averaged_outputs)
                if not self.cfg.TRAIN.TUNE_MODE:
                    pbar.add(len(lfads_input.enc_input))

            # collect the outputs for all batches and split them up into the appropriate variables
            all_outputs = list(zip(*all_outputs)) # transpose the list / tuple
            all_outputs = [np.concatenate(t, axis=0) for t in all_outputs]

            samp_out = self.get_sampling_output(all_outputs, dataset_names, output_file, prefix)
        if not save:
            # If saving is disabled, load from the file and delete it
            output = self.load_posterior_averages(
                self.cfg.TRAIN.MODEL_DIR, merge_tv=merge_tv)
            os.remove(output_file)
            return output
    
    def load_posterior_averages(self, model_dir, merge_tv=False): 
        return load_posterior_averages(model_dir, merge_tv=merge_tv)

    def get_sampling_output(self, outputs, dataset_names, output_file, prefix, save=True): 
        co_means, co_stddevs, factors, gen_states, \
            gen_init, gen_inputs, con_states, behavior, \
            ic_post_mean, ic_post_logvar = outputs
        dataset_names = np.concatenate(dataset_names).astype('str')
        
        rates = self.transform_factors_to_rates(factors, dataset_name=dataset_names)

        # return the output in an organized tuple
        samp_out = SamplingOutput(
            rates=rates,
            factors=factors,
            gen_states=gen_states,
            gen_inputs=gen_inputs,
            gen_init=gen_init,
            ic_post_mean=ic_post_mean,
            ic_post_logvar=ic_post_logvar,
            ic_prior_mean=self.ic_prior_mean.numpy(),
            ic_prior_logvar=self.ic_prior_logvar.numpy(),
            predicted_behavior=behavior)

        if save:
            # writes the output to the a file in the model directory
            with h5py.File(output_file, 'a') as hf:
                output_fields = list(samp_out._fields)
                for field in output_fields:
                    hf.create_dataset(
                        prefix+field,
                        data=getattr(samp_out, field))
            # Save the indices if they exist
            if self.train_inds is not None and self.valid_inds is not None:
                with h5py.File(output_file, 'a') as hf:
                    ds_name = self.ds_names[0]
                    if 'train_inds' not in hf.keys():
                        hf.create_dataset('train_inds', data=self.train_inds[ds_name])
                    if 'valid_inds' not in hf.keys():
                        hf.create_dataset('valid_inds', data=self.valid_inds[ds_name])
                        
        return samp_out    

    def restore_weights(self, lve=True):
        """
        Restores the weights of the model from the most 
        recent or least validation error checkpoint

        lve: bool (optional)
            whether to use the least validation error 
            checkpoint, by default True
        """
        # pass some data through the model to initialize weights
        cfg = self.cfg
        data_shape, _, ext_input_shape, name_shape, beh_shape = self.get_input_shapes(10)
        noise = LFADSInput(
            np.ones(shape=data_shape, dtype=np.float32),
            np.ones(shape=ext_input_shape, dtype=np.float32),
            np.full(shape=name_shape, fill_value=''),
            np.ones(shape=beh_shape, dtype=np.float32)
        )
        self.call(noise)
        if lve:
            # restore the least validation error checkpoint
            self.lve_ckpt.restore(
                self.lve_manager.latest_checkpoint
            ).assert_nontrivial_match()
            self.lgr.info("Restoring the LVE model.")
        else:
            # restore the most recent checkpoint
            self.mrc_ckpt.restore(
                self.mrc_manager.latest_checkpoint
            ).assert_nontrivial_match()
            self.lgr.info("Restoring the most recent model.")
        self.is_trained=True

    def write_weights_to_h5(self, fpath):
        """Writes the weights of the model to an HDF5 file, 
        for directly transferring parameters from LF2 to lfadslite
        
        Parameters
        ----------
        fpath : str
            The path to the HDF5 file for saving the weights.
        """
        with h5py.File(fpath, 'w') as h5file:
            for variable in self.trainable_variables:
                array = variable.numpy()
                h5file.create_dataset(
                    variable.name,
                    array.shape,
                    data=array,
                )
