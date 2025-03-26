import shutil, logging, yaml, io, os, math, copy, h5py, glob
import numpy as np
import pandas as pd
from yacs.config import CfgNode as CN

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import Progbar
from tensorflow.keras.layers import Dense, ReLU, LayerNormalization, GRU
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.optimizers import Adam

import tensorflow_probability as tfp
tfd = tfp.distributions

from umap import UMAP # need numba<0.53
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lfads_tf2.models import LFADS
from lfads_tf2.subclasses.dimreduced.models import DimReducedLFADS
from lfads_tf2.subclasses.behavior.models import BehaviorLFADS
from lfads_tf2.tuples import LoadableData, LFADSInput
from lfads_tf2.utils import load_data, unflatten
from lfads_tf2.subclasses.dimreduced.defaults import get_cfg_defaults as get_dimreduced_lfads_defaults
from lfads_tf2.subclasses.behavior.defaults import get_cfg_defaults as get_behavior_lfads_defaults
from lfads_tf2.initializers import variance_scaling
from lfads_tf2.regularizers import DynamicL2

from align_tf2.utils import rgetattr
from align_tf2.tuples import SingleModelOutput, AlignmentOutput
from align_tf2.layers import MinibatchDiscrimination
from align_tf2.defaults import get_cfg_defaults as get_align_defaults

class AlignLFADS(Model):
    def __init__(self, cfg_node=None, cfg_path=None, align_dir=None, lve=True):
        """A model that creates and trains alignment architectures surrounding LFADS.
        
        Parameters
        ----------
        cfg_path : str, optional
            A path to a YAML alignment configuration file, by default None
        align_dir : str, optional
            A path to a previously trained alignment model, by default None
        """
        super(AlignLFADS, self).__init__()

        # Check args
        num_set = sum([x != None for x in [cfg_node, cfg_path, align_dir]])
        if num_set > 1:
            raise ValueError(
                "Only one of `cfg_node`, `cfg_path`, or `align_dir` may be set")

        if align_dir:
            print(f"Loading aligned model from {align_dir}")
            assert os.path.exists(align_dir), \
                f"No aligned model exists at {align_dir}"
            as_path = os.path.join(align_dir, 'align_spec.yaml')

            cfg = get_align_defaults()
            cfg.merge_from_file(as_path)
            cfg.freeze()
            self.load_from_config(cfg)
            self.is_trained = True
            self.from_existing = True
        else:
            print("Initializing new aligner")
            cfg = get_align_defaults()
            if cfg_node:
                cfg.merge_from_other_cfg(cfg_node)
            elif cfg_path:
                cfg.merge_from_file(cfg_path)
            else:
                print("Using default config")
            cfg.freeze()

            # check that align_dir overwrite is handled properly
            align_dir = cfg.TRAIN.ALIGN_DIR
            if not cfg.TRAIN.NI_MODE:
                if cfg.TRAIN.OVERWRITE:
                    if os.path.exists(align_dir):
                        print("Overwriting align directory")
                        shutil.rmtree(align_dir)
                else:
                    assert not os.path.exists(align_dir), \
                        (f"An aligned model already exists at `{align_dir}`. "
                        "Load it by explicitly providing the path, or specify a new `align_dir`.")
                os.makedirs(align_dir)
            as_path = os.path.join(align_dir, 'align_spec.yaml')
            with open(as_path, 'w') as as_file:
                as_file.write(cfg.dump())
            os.chmod(as_path, 0o775)

            # make a copy of the lfads model in the align directory
            lfads_dir = os.path.join(cfg.TRAIN.ALIGN_DIR, 'pbt_model')
            shutil.copytree(cfg.TRAIN.MODEL_DIR, lfads_dir)
            os.chmod(lfads_dir, 0o775)
            # adjust the lfads config for alignment
            lfads_cfg_path = os.path.join(lfads_dir, 'model_spec.yaml')
            if cfg.MODEL.DAY0_MODEL_TYPE == 'dimreduced': 
                lfads_cfg = get_dimreduced_lfads_defaults()
            elif cfg.MODEL.DAY0_MODEL_TYPE == 'behavior':
                    lfads_cfg = get_behavior_lfads_defaults()
            lfads_cfg.merge_from_file(lfads_cfg_path)
            align_mode_cfg = CN(unflatten({
                'TRAIN.MODEL_DIR': lfads_dir}))
            lfads_cfg.merge_from_other_cfg(align_mode_cfg)
            # update posterior sampling 
            lfads_cfg.MODEL.SAMPLE_POSTERIORS = cfg.MODEL.SAMPLE_POSTERIORS
            # update KL parameters 
            lfads_cfg.TRAIN.KL.START_EPOCH = cfg.TRAIN.KL.START_EPOCH
            lfads_cfg.TRAIN.KL.INCREASE_EPOCH = cfg.TRAIN.KL.INCREASE_EPOCH
            lfads_cfg.TRAIN.KL.IC_WEIGHT = cfg.TRAIN.KL.IC_WEIGHT
            lfads_cfg.TRAIN.KL.CO_WEIGHT = cfg.TRAIN.KL.CO_WEIGHT
            # Overwrite the updated config
            with open(lfads_cfg_path, 'w') as cfg_file:
                cfg_file.write(lfads_cfg.dump())
            os.chmod(lfads_cfg_path, 0o775)

            self.load_from_config(cfg)
            self.is_trained = False
            self.from_existing = False

        # ===== CREATE TensorBoard SUMMARY_WRITER =====
        if cfg.TRAIN.USE_TB:
            self.summary_writer = tf.summary.create_file_writer(cfg.TRAIN.ALIGN_DIR)

        # set up logging using the configuration file in this directory
        log_conf_path = os.path.join(os.path.dirname(__file__), 'logging_conf.yaml')
        logging_conf = yaml.full_load(open(log_conf_path))
        os.chmod(log_conf_path, 0o775)
        # tell the file handler where to write output
        logging_conf['handlers']['logfile']['filename'] = os.path.join(align_dir, 'train.log')
        logging_conf['handlers']['csv']['filename'] = os.path.join(align_dir, 'train_data.csv')
        # set up logging from the configuation dict and create a global logger
        logging.config.dictConfig(logging_conf)
        self.lgr = logging.getLogger('align')
        self.csv_lgr = logging.getLogger('train_csv')
        os.chmod(logging_conf['handlers']['logfile']['filename'], 0o775)
        os.chmod(logging_conf['handlers']['csv']['filename'], 0o775)

        # ===== METRICS and LOGGING =====
        if cfg.MODEL.TYPE == 'FFN-KL':
            self.logging_metrics = [
                'epoch',
                'loss',
                'kl',
                'nll',
                'val_loss',
                'val_kl',
                'val_nll',
                'gnorm',
                'kl_gnorm',
                'nll_gnorm',
                'lr',
                'nll_weight',
            ]
        else:
            raise AssertionError(f"Model type {cfg.MODEL.TYPE} not recognized.")

        # create the CSV header if this is a new train_data.csv
        csv_log = logging_conf['handlers']['csv']['filename']
        if os.stat(csv_log).st_size == 0:
            self.csv_lgr.info(','.join(self.logging_metrics))

        # load the training dataframe
        self.train_df = pd.read_csv(csv_log, index_col='epoch')

        # create the metrics for saving data during training
        self.all_metrics = {name: tf.keras.metrics.Mean() for name in self.logging_metrics}

        # ===== Initialize Training Params =====
        self.cur_epoch = tf.Variable(0, trainable=False)
        self.cur_patience = tf.Variable(0, trainable=False)
        self.loss_scale = tf.Variable(cfg.TRAIN.LOSS_SCALE, trainable=False)
        self.nll_ramping_weight = tf.Variable(0.0, trainable=False)

        # ===== AUTOGRAPH FUNCTIONS =====
        data_shape = [
            None, self.cfg.MODEL.SEQ_LEN, self.cfg.MODEL.DATA_DIM]
        self._graph_step = tf.function(
            func=self._step,
            input_signature=[
                tf.TensorSpec(shape=data_shape),
                tf.TensorSpec(shape=data_shape),
                tf.TensorSpec(shape=None, dtype=tf.bool),
                tf.TensorSpec(shape=None, dtype=tf.bool)])

        if self.from_existing:
            self.restore_weights(lve=lve)

    def load_from_config(self, cfg):
        """Creates the models and alignment objects based on the configNode.
        
        Parameters
        ----------
        cfg : yacs.CfgNode
            A configuration object that stores architecture and training instructions.
        """
        self.cfg = cfg
        # load and freeze the lfads model
        lfads_dir = os.path.join(cfg.TRAIN.ALIGN_DIR, 'pbt_model')
        if cfg.MODEL.DAY0_MODEL_TYPE == 'dimreduced':
            self.lfads_day0 = DimReducedLFADS(model_dir=lfads_dir)
            self.lfads_dayk = AlignedDimReducedLFADS(model_type=cfg.MODEL.TYPE,
                                    hidden_dim=cfg.MODEL.HIDDEN_DIM,
                                    model_dir=lfads_dir)
        elif cfg.MODEL.DAY0_MODEL_TYPE == 'behavior':
            self.lfads_day0 = BehaviorLFADS(model_dir=lfads_dir)
            self.lfads_dayk = AlignedBehaviorLFADS(model_type=cfg.MODEL.TYPE,
                                    hidden_dim=cfg.MODEL.HIDDEN_DIM,
                                    model_dir=lfads_dir)
                
        # create alignment optimizer
        self.lfads_dayk.optimizer = Adam(learning_rate=cfg.TRAIN.LR.INIT,
                                         epsilon=cfg.TRAIN.ADAM_EPSILON)
        
        # make everything not trainable     
        self.lfads_day0.trainable = False
                
        dayk_trainables = cfg.MODEL.DAYK_LFADS_TRAINABLES
        self.lfads_dayk.trainable = False

        if dayk_trainables:
            # a separate storage for trainable vars
            self.trainable_vars = []
            # set the specified attributes to trainable
            for attr_name in dayk_trainables:
                layer = rgetattr(self.lfads_dayk, attr_name)
                assert issubclass(type(layer), tf.keras.layers.Layer), \
                    f"`{attr_name} in `DAYK_LFADS_TRAINABLES` must be a subclass of `tf.keras.layers.Layer`."
                layer.trainable = True
                [self.trainable_vars.append(v) for v in layer.trainable_variables]

        # turn off dropout for the LFADS models
        self.lfads_day0.training.assign(False)
        self.lfads_dayk.training.assign(False)
        # update lowd_readin L2 scale
        readin_regularizer = self.lfads_dayk.lowd_readin.layers[0].kernel_regularizer
        if readin_regularizer is not None:
            readin_regularizer.scale.assign(cfg.TRAIN.L2.READIN_SCALE)

        # ===== CREATE THE CHECKPOINTS =====
        ckpt_dir = os.path.join(cfg.TRAIN.ALIGN_DIR, 'align_ckpts')
        # checkpointing for least validation error model
        self.lve_ckpt = tf.train.Checkpoint(
            aligner=self.lfads_dayk.generator, lfads=self.lfads_dayk)
        lve_ckpt_dir = os.path.join(ckpt_dir, 'least_val_err')
        self.lve_manager = tf.train.CheckpointManager(
                self.lve_ckpt, directory=lve_ckpt_dir,
                max_to_keep=1, checkpoint_name='lve-ckpt')
        # checkpointing for the most recent model
        self.mrc_ckpt = tf.train.Checkpoint(
            aligner=self.lfads_dayk.generator, lfads=self.lfads_dayk)
        mrc_ckpt_dir = os.path.join(ckpt_dir, 'most_recent')
        self.mrc_manager = tf.train.CheckpointManager(
            self.mrc_ckpt, directory=mrc_ckpt_dir,
            max_to_keep=1, checkpoint_name='mrc-ckpt')

    def call(self, data, training=False, use_logrates=False):
        """Defines the forward pass through the alignment network.
        
        Parameters
        ----------
        data : np.ndarray or tf.Tensor
            Data should be the same as the typical input to LFADS - a tensor of 
            dimensions batch_size x seq_len x num_neurons.
        training : bool, optional
            Whether this call is for training (indicates whether to use dropout), 
            by default False
        logrates : bool, optional
            Whether to return logrates, which are more numerically stable for 
            loss calculations, by default False
        
        Returns
        -------
        np.ndarray
            A tensor of the estimated rates for the spikes in `data`.
        tfd.MultivariateNormalDiag
            The distribution of initial conditions of the generator.
        tfd.MultivariateNormalDiag
            The distribution of controller outputs.
        """
        data_name = ''
        lfads_input = LFADSInput(
            enc_input=data,
            # external inputs are not currently supported
            ext_input=tf.zeros(tf.concat([tf.shape(data)[:-1], [0]], axis=0)),
            dataset_name=tf.fill((tf.shape(data)[0],), data_name),
            behavior=np.zeros(shape=tf.shape(data)[0], dtype=np.float32))
        outputs = self.lfads_dayk(lfads_input, use_logrates=use_logrates)
        return outputs

    def load_datasets(self, align_input=None):
        """Saves data arrays and tf.data.Datasets to the alignLFADS 
        object for use during training.
        
        Parameters
        ----------
        align_input : align_tf2.tuples.AlignInput, optional
            A namedtuple of numpy arrays containing training and 
            validation data for day-0 and day-k. See definition
            for more details. By default, None causes the model
            to load data from the directories specified in the 
            config.

        """

        if align_input is None:
            # if alignment output is not given, load from config
            day0_data_dir = self.cfg.TRAIN.DATA.DAY0
            self.lgr.info(f"Loading day-0 data from {day0_data_dir}.")
            d0_train_data, d0_valid_data = load_data(day0_data_dir)[0]
            d0_train_inds, d0_valid_inds = load_data(day0_data_dir, signal='inds')[0]
            dayk_data_dir = self.cfg.TRAIN.DATA.DAYK
            self.lgr.info(f"Loading day-k data from {dayk_data_dir}.")
            dk_train_data, dk_valid_data = load_data(dayk_data_dir)[0]
            dk_train_inds, dk_valid_inds = load_data(dayk_data_dir, signal='inds')[0]
        else:
            # if alignment input is given, use the arrays
            self.lgr.info(f"Using datasets passed as arguments to `load_datasets`.")
            d0_train_data, d0_valid_data, dk_train_data, dk_valid_data, \
                d0_train_inds, d0_valid_inds, dk_train_inds, dk_valid_inds = align_input

        # save the indices
        self._d0_train_inds = d0_train_inds
        self._d0_valid_inds = d0_valid_inds
        self._dk_train_inds = dk_train_inds
        self._dk_valid_inds = dk_valid_inds
        # save the arrays of data
        self._d0_train_data = tf.cast(d0_train_data, tf.float32)
        self._d0_valid_data = tf.cast(d0_valid_data, tf.float32)
        self._dk_train_data = tf.cast(dk_train_data, tf.float32)
        self._dk_valid_data = tf.cast(dk_valid_data, tf.float32)
        # create tensorflow datasets
        self._d0_train_ds = tf.data.Dataset.from_tensor_slices(self._d0_train_data)
        self._d0_valid_ds = tf.data.Dataset.from_tensor_slices(self._d0_valid_data)
        self._dk_train_ds = tf.data.Dataset.from_tensor_slices(self._dk_train_data)
        self._dk_valid_ds = tf.data.Dataset.from_tensor_slices(self._dk_valid_data)

        # ===== SET UP UMAP =====
        if self.cfg.TRAIN.USE_TB:
            # set up the UMAP dimensionality reduction, fit to day0, and calculate day0 embedding
            def get_align_point_cloud(data):
                align_target = self.get_align_tensors(data)
                align_point_cloud = tf.reshape(align_target, (-1, align_target.shape[-1]))
                return align_point_cloud
            # get the point cloud for alignment
            d0_train_align_point_cloud = get_align_point_cloud(self._d0_train_data)
            d0_valid_align_point_cloud = get_align_point_cloud(self._d0_valid_data)
            # fit UMAP to the training data and transform training and validation data
            self.reducer = UMAP()
            self.lgr.info('Fitting and transforming day0 with UMAP.')
            self.d0_train_embed = self.reducer.fit_transform(d0_train_align_point_cloud)
            self.d0_valid_embed = self.reducer.transform(d0_valid_align_point_cloud)

    def smart_init(self):
        if self.lfads_dayk.cfg.MODEL.NORMALIZE: 
            print('Setting Day K normalization parameters...')
            norm_param_path = os.path.join(self.cfg.TRAIN.ALIGN_DIR, 'normalization_dayk.h5')
            f = h5py.File(norm_param_path, 'r')
            bias = f['bias'][()]
            mat = f['matrix'][()]
            f.close()

            self.lfads_dayk.load_matrix_to_lowd_readin(mat, bias=bias, freeze=True, norm_layer=True)
            
    def get_align_tensors(self, d0_data):
        """Returns the tensors that we are aligning, based on the configuration.
        
        Parameters
        ----------
        d0_data : np.ndarray or tf.Tensor
            Data should be the same as the typical input to LFADS - a tensor of 
            dimensions batch_size x seq_len x num_neurons.
        
        Returns
        -------
        tf.Tensor
            The tensor to be aligned - shape will be different depending on
            which tensor is being selected.
        """
        data_name = ''
        lfads_input = LFADSInput(
            enc_input=d0_data,
            # external inputs not currently supported.
            ext_input=tf.zeros(tf.concat([tf.shape(d0_data)[:-1], [0]], axis=0)),
            dataset_name=tf.fill((tf.shape(d0_data)[0],), data_name),
            behavior=tf.zeros(tf.shape(d0_data)[0]))
        output = self.lfads_day0(lfads_input)
        return [getattr(output, align_data) for align_data in self.cfg.MODEL.ALIGN_DATA]

    def _step(self, d0_batch, dk_batch, validation=False):
        """A step that performs loss calculations and gradient decent for 
        training and validation
        
        Parameters
        ----------
        d0_batch : tf.Tensor
            A batch of day-0 data, should be the same as the typical input to LFADS - a
            tensor of dimensions batch_size x seq_len x num_neurons.
        dk_batch : tf.Tensor
            A batch of day-k data, should be the same as the typical input to LFADS - a 
            tensor of dimensions batch_size x seq_len x num_neurons.
        validation : bool, optional
            Whether to perform validation on this step (as opposed to validation), 
            by default False
        """
        cfg = self.cfg

        def kl_cost(d0_align, dk_align):
            """ Computes KL cost, assuming multivariate gaussian with full covariance """

            # add jitter to diagonal 
            jitter_mat = tf.eye(d0_align.shape[-1]) * 1.e-6

            if tf.rank(d0_align) > 2:
                align_dim = tf.shape(d0_align)[-1]
                d0_align = tf.reshape(d0_align, (-1, align_dim))
                dk_align = tf.reshape(dk_align, (-1, align_dim))                

            d0_mean = tf.reduce_mean(d0_align, axis=0)
            dk_mean = tf.reduce_mean(dk_align, axis=0)
            d0_cov = tfp.stats.covariance(d0_align)
            dk_cov = tfp.stats.covariance(dk_align)
            d0_dist = tfd.MultivariateNormalFullCovariance(loc=d0_mean, covariance_matrix=d0_cov+jitter_mat)
            dk_dist = tfd.MultivariateNormalFullCovariance(loc=dk_mean, covariance_matrix=dk_cov+jitter_mat)

            return tfd.kl_divergence(d0_dist, dk_dist)

        def neg_log_likelihood(data, logrates):
            """ Computes the log likelihood of the data, given current predicted spiking rates. """
            nll_all = tf.nn.log_poisson_loss(data, logrates, compute_full_loss=True)
            return tf.reduce_mean(nll_all)

        def cross_entropy(target_label, logits):
            """ computes binary cross entropy loss """
            if target_label:
                label = 1.0
            else:
                label = 0.0
            labels = label * tf.ones_like(logits)
            ce_all = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
            return tf.reduce_mean(ce_all)

        def train_step(d0_batch, dk_batch):
            """ Performs a step of optimization. """

            # get the day 0 alignment target tensor
            d0_align_target = self.get_align_tensors(d0_batch)

            # align using a FFN with KL cost
            if cfg.MODEL.TYPE == 'FFN-KL':
                with tf.GradientTape(persistent=True) as tape:
                    # send the data through the alignment model
                    outputs = self.call(dk_batch, use_logrates=True)
                    nll = neg_log_likelihood(dk_batch, outputs.rates)
                    dk_align = [getattr(outputs, align_data) for align_data in cfg.MODEL.ALIGN_DATA]
                    # use the FFN-KL model
                    kl = 0
                    for ii in range(len(d0_align_target)):
                        kl += kl_cost(d0_align_target[ii], dk_align[ii])  * cfg.TRAIN.ALIGN_LOSS_WEIGHT[ii]

                    nll_loss = nll * self.nll_ramping_weight * cfg.TRAIN.NLL.WEIGHT
                    lfads_loss = nll_loss

                    if 'lowd_readin' in cfg.MODEL.DAYK_LFADS_TRAINABLES:
                        lfads_loss += tf.reduce_sum(
                            self.lfads_dayk.lowd_readin.losses)
                    
                    if self.cfg.MODEL.SAMPLE_POSTERIORS: 
                        # add KL cost 
                        kl_post = self.lfads_dayk.weighted_kl_loss(
                            outputs.ic_means, outputs.ic_stddevs, 
                            outputs.co_means, outputs.co_stddevs
                        )
                        lfads_loss += kl_post

                    # TODO: update this if we switch to graph mode
                    loss = (kl +
                            lfads_loss * cfg.TRAIN.LFADS_LOSS_WEIGHT)

                    scaled_loss = self.loss_scale * loss
                    
                kl_grads = tape.gradient(kl, self.trainable_vars)
                kl_grads, kl_gnorm = tf.clip_by_global_norm(kl_grads, cfg.TRAIN.MAX_GRAD_NORM)
                nll_grads = tape.gradient(nll_loss, self.trainable_vars)
                nll_grads, nll_gnorm = tf.clip_by_global_norm(nll_grads, cfg.TRAIN.MAX_GRAD_NORM)

                # perform an optimization step
                grads = tape.gradient(scaled_loss, self.trainable_vars)
                grads, gnorm = tf.clip_by_global_norm(grads, cfg.TRAIN.MAX_GRAD_NORM)

                self.lfads_dayk.optimizer.apply_gradients(zip(grads, self.trainable_vars))
                # record the metrics
                metric_values = {
                    'gnorm': gnorm,
                    'loss': loss,
                    'kl': kl,
                    'nll': nll,
                    'kl_gnorm': kl_gnorm,
                    'nll_gnorm': nll_gnorm
                }

            else:
                raise AssertionError(f"Model type {cfg.MODEL.TYPE} not recognized.")

            for name, value in metric_values.items():
                self.all_metrics[name].update_state(value)

        def val_step(d0_batch, dk_batch):
            """Computes the validation metrics on a batch of day-0 data and a batch of day-k data.
            """
            # get the day 0 alignment target tensor
            d0_align_target = self.get_align_tensors(d0_batch)
            outputs = self.call(dk_batch, use_logrates=True)
            val_nll = neg_log_likelihood(dk_batch, outputs.rates)
            dk_align = [getattr(outputs, align_data) for align_data in cfg.MODEL.ALIGN_DATA]
            if cfg.MODEL.TYPE == 'FFN-KL':
                val_kl = 0
                for ii in range(len(d0_align_target)):
                    val_kl += kl_cost(d0_align_target[ii], dk_align[ii]) * cfg.TRAIN.ALIGN_LOSS_WEIGHT[ii]

                # TODO: update this if we switch to graph mode
                val_loss = val_kl + val_nll * self.nll_ramping_weight * cfg.TRAIN.NLL.WEIGHT

                # record training statistics
                metric_values = {
                    'val_loss': val_loss,
                    'val_kl': val_kl,
                    'val_nll': val_nll,
                }
            
            else:
                raise AssertionError(f"Model type {cfg.MODEL.TYPE} not recognized.")

            for name, value in metric_values.items():
                self.all_metrics[name].update_state(value)

        if tf.reduce_any(validation):
            val_step(d0_batch, dk_batch)
        else:
            train_step(d0_batch, dk_batch)

    def update_learning_rate(self, optimizer, lr_metric):
        """Updates the learning rate of the optimizer."""
        cfg = self.cfg.TRAIN.LR
        prev_epoch = len(self.train_df)
        # if the config has just been updated, just assign the new initial learning rate
        if 0 < cfg.DECAY < 1 and prev_epoch > cfg.PATIENCE:
            epochs_since_decay = (self.train_df.lr == self.train_df.at[prev_epoch, 'lr']).sum()
            if epochs_since_decay >= cfg.PATIENCE:
                # compare the current lr_metric to the max over a window of previous epochs
                winmax_lr_metric = self.train_df.iloc[-(cfg.PATIENCE+1):-1][lr_metric].max()
                cur_lr_metric = self.train_df.at[prev_epoch, lr_metric]
                # if the current lr_metric is greater than the max in the window, decay LR
                if cur_lr_metric > winmax_lr_metric:
                    new_lr = max([cfg.DECAY * optimizer.lr.numpy(), cfg.STOP])
                    optimizer.lr.assign(new_lr)
        return optimizer.lr.numpy()

    def update_ramping_weights(self):
        """Updates the ramping weight variables. 
        
        In order to allow the model to learn more quickly, we introduce 
        regularization slowly by linearly increasing the ramping weight 
        as the model is training, to maximum ramping weights of 1. This 
        function computes the desired weight and assigns it to the ramping 
        variables, which are later used in the NLL calculations.
        Ramping is determined by START_EPOCH, or the epoch to start 
        ramping, and INCREASE_EPOCH, or the number of epochs over which 
        to increase to full regularization strength. Ramping can occur 
        separately for NLL losses, based on their respective 
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
        nll_weight = compute_weight(cfg.NLL.START_EPOCH, cfg.NLL.INCREASE_EPOCH)
        # assign values to the ramping weight variables
        self.nll_ramping_weight.assign(nll_weight)
        # report the current KL and L2 weights to the metrics
        self.all_metrics['nll_weight'].update_state(nll_weight)

    def train_epoch(self):
        """Trains the alignment model for a single epoch."""
        cfg = self.cfg
        self.cur_epoch.assign_add(1)
        cur_epoch = self.cur_epoch.numpy()

        self.update_ramping_weights()

        if cfg.MODEL.TYPE == 'FFN-KL':
            # update the learning rate for the aligner
            if cfg.TRAIN.EARLY_STOPPING_COST == 'NLL-KL': 
                lr_metric = 'val_loss'
            elif cfg.TRAIN.EARLY_STOPPING_COST == 'NLL': 
                lr_metric = 'val_nll'
            cur_lr = self.update_learning_rate(self.lfads_dayk.optimizer, lr_metric)
            epoch_header_template = ', '.join([
                "Epoch {epoch}/{max_epochs}",
                "Patience: {patience}",
                "LR: {cur_lr:.2E}",
            ])
            epoch_header = epoch_header_template.format(**{
                'epoch': cur_epoch,
                'max_epochs' : cfg.TRAIN.MAX_EPOCHS - 1,
                'patience': self.cur_patience.numpy(),
                'cur_lr': cur_lr,
            })
            self.lgr.info(epoch_header)
        
        def prep_dataset(data, dataset, is_train, is_day0):
            # get the number of training samples for this dataset
            n_samples = data.shape[0]
            # get the batch size for this dataset
            train_bs = cfg.TRAIN.BATCH_SIZE
            valid_bs = cfg.TRAIN.VALID_BATCH_SIZE
            batch_size = train_bs if (is_train or valid_bs < 0) \
                else valid_bs
            # drop incomplete batches if there are more than one
            drop_remainder = n_samples > batch_size
            if is_train:
                # shuffle training data
                dataset = dataset.shuffle(n_samples)
            # batch all data
            dataset = dataset.batch(batch_size, drop_remainder)
            if is_day0:
                # loop indefinitely on day0 data
                dataset = dataset.repeat()
            return dataset

        # Prepare all of the datasets for training
        d0_train_ds = prep_dataset(
            self._d0_train_data, self._d0_train_ds, True, True)
        dk_train_ds = prep_dataset(
            self._dk_train_data, self._dk_train_ds, True, False)
        d0_valid_ds = prep_dataset(
            self._d0_valid_data, self._d0_valid_ds, False, True)
        dk_valid_ds = prep_dataset(
            self._dk_valid_data, self._dk_valid_ds, False, False)

        # create a progress bar for the training steps
        num_steps = sum(1 for _ in dk_train_ds)
        pbar = Progbar(num_steps, unit_name='step')

        # perform a step of training and validation
        for d0_batch, dk_batch in zip(d0_train_ds, dk_train_ds):
            # train the aligner / generator
            self._graph_step(d0_batch, dk_batch)
            pbar.add(1)
        
        for d0_batch, dk_batch in zip(d0_valid_ds, dk_valid_ds):
            self._graph_step(d0_batch, dk_batch, validation=True)
        # collect the results
        results = {name: m.result().numpy() for name, m in self.all_metrics.items()}
        results.update({
            'epoch': cur_epoch,
            'lr': cur_lr,
        })
        _ = [m.reset_states() for name, m in self.all_metrics.items()]
        if cfg.MODEL.TYPE == 'FFN-KL':
            train_template = " - ".join([
                "    loss: {loss:.3f}",
                "    kl: {kl:.3f}",
                "    nll: {nll:.3f}",
                "gnorm: {gnorm:.3f}",
                "kl_gnorm: {kl_gnorm:.3f}",
                "nll_gnorm: {nll_gnorm:.3f}"])
            val_template = " - ".join([
                "val_loss: {val_loss:.3f}",
                "val_kl: {val_kl:.3f}",
                "val_nll: {val_nll:.3f}"])
        
        
        self.lgr.info(train_template.format(**results))
        self.lgr.info(val_template.format(**results))

        # write the metrics and HPs to the in-memory `train_df` and to the CSV file
        new_results_df = pd.DataFrame({key: [val] for key, val in results.items()}).set_index('epoch')
        self.train_df = pd.concat([self.train_df, copy.deepcopy(new_results_df)])
        # add quotation marks so commas inside strings are ignored by csv parser
        convert_for_csv = lambda data: f'\"{data}\"' if type(data) == str else str(data)
        csv_output = [convert_for_csv(results[log_metric]) for log_metric in self.logging_metrics]
        self.csv_lgr.info(','.join(csv_output))

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
            if np.isnan(loss) or np.isnan(val_loss):
                self.lgr.info("STOPPING: NaN found in loss.")
                pass_check = False
            return pass_check

        def check_lr(train_df):
            """ Check if training should stop because of the learning rate. """
            cfg = self.cfg.TRAIN
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
            cfg = self.cfg.TRAIN
            if cur_epoch > cfg.NLL.INCREASE_EPOCH:
                if cfg.EARLY_STOPPING_COST == 'NLL-KL': 
                    best_epoch = train_df.val_loss[train_df.index > cfg.NLL.INCREASE_EPOCH].idxmin()
                elif cfg.EARLY_STOPPING_COST == 'NLL':
                    best_epoch = train_df.val_nll[train_df.index > cfg.NLL.INCREASE_EPOCH].idxmin()
                # save a checkpoint if this model is the best and beyond `self.last_ramp_epoch`
                # use the `self.train_status` to report the status of early stopping
                if best_epoch == cur_epoch:
                    # save a checkpoint if this model is the best
                    self.lve_manager.save()
                # stop training if `smth_val_nll` does not improve  after `cfg.PATIENCE` epochs
                
                self.cur_patience.assign(max([cur_epoch - best_epoch, 0]))
                if self.cur_patience.numpy() >= cfg.PATIENCE:
                    self.lgr.info(f"STOPPING: No improvement in `val_loss` for {cfg.PATIENCE} epochs.")
                    pass_check = False
            return pass_check

        # Run all of the checks for stopping criterion
        check_funcs = [check_max_epochs, check_nans, check_lr, check_earlystop]
        checks_passed = [check_func(self.train_df) for check_func in check_funcs]
        # end the training loop if not all checks passed
        if not all(checks_passed):
            results['done'] = True
        else:
            self.mrc_manager.save()

        if self.cfg.TRAIN.USE_TB:
            # plot the ALIGNED TENSORS IN TENSORBOARD
            def make_figure(d0_embed, dk_data):
                output = self.call(dk_data)
                dk_align = getattr(output, cfg.MODEL.ALIGN_DATA)
                dk_align_point_cloud = tf.reshape(dk_align, (-1, dk_align.shape[-1]))
                figure = plt.figure(figsize=(10, 10))
                self.lgr.info(f"Applying UMAP to {cfg.MODEL.ALIGN_DATA}.")
                dk_embed = self.reducer.transform(dk_align_point_cloud)
                plt.scatter(d0_embed[:,0], d0_embed[:,1], s=1, alpha=0.3, label='day0')
                plt.scatter(dk_embed[:,0], dk_embed[:,1], s=1, alpha=0.3, label='dayk')
                plt.legend()
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

            # log the loss for tensorboard
            with self.summary_writer.as_default():
                tf.summary.experimental.set_step(cur_epoch)
                with tf.name_scope('align'):
                    if cur_epoch %5 == 0 or cur_epoch == 1:
                        train_fig = make_figure(self.d0_train_embed, self._dk_train_data)
                        tf.summary.image(
                            f'train_{cfg.MODEL.ALIGN_DATA}_umap',
                            figure_to_tf(train_fig))
                        valid_fig = make_figure(self.d0_valid_embed, self._dk_valid_data)
                        tf.summary.image(
                            f'valid_{cfg.MODEL.ALIGN_DATA}_umap',
                            figure_to_tf(valid_fig))
                    for key, val in results.items():
                        tf.summary.scalar(key, val)
                    tf.summary.scalar('patience', self.cur_patience.numpy())

            self.summary_writer.flush()

        # remove NaN's, which can cause bugs for TensorBoard
        results = {key: val for key, val in results.items() if val != np.nan}
        return results


    def sample_and_average(self,
                           aligned_model=True,
                           dayk_data=True,
                           n_samples=50,
                           batch_size=64,
                           save=True,
                           merge_tv=False,
                           lve=False,
                           ps_filename='posterior_samples.h5',
                           restore_weights=True):
        """Performs posterior sampling to estimate rates.

        This function will use whatever data has been 
        loaded into the model using the `load_datasets` 
        function to perform posterior sampling.
        
        Parameters
        ----------
        aligned_model : bool, optional
            Whether to use the aligned model instead of 
            the original model, by default True.
        n_samples : int, optional
            The number of samples to take from each initial 
            condition and controller output distribution, 
            by default 50
        batch_size : int, optional
            The batch size of the posterior sampling 
            operation, by default 256
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

        # Restore weights for the appropriate models
        if restore_weights:
            self.restore_weights(lve=lve)

        if self.lfads_day0.cfg.MODEL.SAMPLE_POSTERIORS:
            self.lgr.info('Sampling and averaging.')
        else: 
            self.lgr.info('Getting posterior means.')
        # Get the appropriate datasets
        train_ds = self._dk_train_ds if dayk_data else self._d0_train_ds
        valid_ds = self._dk_valid_ds if dayk_data else self._d0_valid_ds
        # Get the appropriate indices
        train_inds = self._dk_train_inds if dayk_data else self._d0_train_inds
        valid_inds = self._dk_valid_inds if dayk_data else self._d0_valid_inds

        def get_lfads_input(dataset):
            """Iterate through a dataset and convert from neural 
            data to LFADS input, so we can use the LFADS posterior
            sampling function.
            """
            lfads_inputs = []
            for batch in dataset.batch(batch_size):
                lfads_input = batch
                lfads_inputs.append(lfads_input)
            return tf.concat(lfads_inputs, axis=0)

        # if DAY0 field is used, use that
        if self.cfg.TRAIN.DATA.DAY0 != '':
            dataset_pathway_name = self.cfg.TRAIN.DATA.DAY0
        # otherwise, use first day
        else:
            dataset_pathway_name = self.lfads_day0.ds_names[0]

        # Compute the input to LFADS for the available datasets
        loadable_data = LoadableData(
            train_data={dataset_pathway_name: get_lfads_input(train_ds)},
            valid_data={dataset_pathway_name: get_lfads_input(valid_ds)},
            train_ext_input=None,
            valid_ext_input=None,
            train_inds={dataset_pathway_name: train_inds},
            valid_inds={dataset_pathway_name: valid_inds},
            train_behavior=None,
            valid_behavior=None)
        # Perform posterior sampling on the aligned data
        model = self.lfads_dayk if aligned_model else self.lfads_day0
        output = model.sample_and_average(
            loadable_data=loadable_data,
            n_samples=n_samples,
            batch_size=batch_size,
            save=save,
            merge_tv=merge_tv,
            ps_filename=ps_filename)
        # Return posterior sampling output if it isn't being saved
        if not save:
            return output

    def restore_weights(self, lve=True):
        """Restores the checkpointed weights of the model.
        
        Parameters
        ----------
        lve : bool
            Whether to use the checkpoint from the model with
            the lowest validation error, as opposed to the most
            recent model. By default, True.
        """

        # first, restore the checkpoints of the LFADS models
        self.lfads_day0.restore_weights(lve=lve)
        self.lfads_dayk.restore_weights(lve=lve)

        if self.is_trained:
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

class Decoder(Model):
    def __init__(self, out_dim, reg):
        super(Model, self).__init__()
        self.linear = Dense(
            out_dim,
            kernel_regularizer=regularizers.l2(reg)
        )
    def call(self, data):
        return self.linear(data)


class AlignedBehaviorLFADS(BehaviorLFADS):
    """Prototype of LFADS model with low-D read-in and alignment network."""
    def __init__(self, model_type='FFN_KL', hidden_dim=50, **kwargs):
        """Creates an alignment network in addition to LFADS."""
        super(AlignedBehaviorLFADS, self).__init__(**kwargs)

    # Model building is in this function in the superclass for proper overloadingDAY0_MODEL_TYPE
    def build_model_from_init_call(self, model_type='FFN-KL', hidden_dim=50, **kwargs):
        super(AlignedBehaviorLFADS, self).build_model_from_init_call(**kwargs)

        # create the alignment layers
        self.generator = Sequential([
            Dense(hidden_dim, kernel_initializer='identity', activation='relu'),
            Dense(hidden_dim, kernel_initializer='identity', activation='relu'),
            Dense(hidden_dim, kernel_initializer='identity'),
            GroupNormalization(groups=1, axis=-1, epsilon=1e-12),
        ])

    def call(self, lfads_input, use_logrates=tf.constant(False)):
        """Passes data through low-dim readin and aligner
            overloads the call function from the base class, then calls it after
            setting the values of the encoder input

        Parameters
        ----------
        lfads_input : lfads_tf2.tuples.LFADSInput
            A namedtuple of tensors containing the data, external inputs, and encoder inputs.
        """
        data = lfads_input.enc_input
        if self.cfg.MODEL.NORMALIZE:
            data = self.norm_layer(data)
        lowd_data = self.lowd_readin(data)
        aligned_data = self.generator(lowd_data)
        lfads_input_out = LFADSInput(
            enc_input = aligned_data,
            ext_input = lfads_input.ext_input,
            dataset_name = lfads_input.dataset_name,
            behavior = lfads_input.behavior)
        # need to use grandparent call function here, so that lfads_input
        # is not overwritten by DimReducedLFADS call
        return super(BehaviorLFADS, self).call(lfads_input_out, use_logrates)

    def posterior_sample_and_average_call(self, input_data, n_samples):
        data = input_data.enc_input
        if self.cfg.MODEL.NORMALIZE:
            data = self.norm_layer(data)
        lowd_data = self.lowd_readin(data)
        aligned_data = self.generator(lowd_data)
        lfads_input = LFADSInput(
            enc_input = aligned_data,
            ext_input = input_data.ext_input,
            dataset_name = input_data.dataset_name,
            behavior = input_data.behavior)
        # need to use grandparent call function here, so that lfads_input
        # is not overwritten by DimReducedLFADS call
        return super(BehaviorLFADS, self).posterior_sample_and_average_call(lfads_input, n_samples)


class AlignedDimReducedLFADS(DimReducedLFADS):
    """Prototype of LFADS model with low-D read-in and alignment network."""
    def __init__(self, model_type='FFN_KL', hidden_dim=50, **kwargs):
        """Creates an alignment network in addition to LFADS."""
        self.hidden_dim = hidden_dim
        super(AlignedDimReducedLFADS, self).__init__(**kwargs)
        
    # Model building is in this function in the superclass for proper overloading
    def build_model_from_init_call(self, model_type='FFN-KL', **kwargs):
        hidden_dim = self.hidden_dim
        
        super(AlignedDimReducedLFADS, self).build_model_from_init_call(**kwargs)

        # create the alignment layers
        self.generator = Sequential([
            Dense(hidden_dim, kernel_initializer='identity', activation='relu'),
            Dense(hidden_dim, kernel_initializer='identity', activation='relu'),
            Dense(hidden_dim, kernel_initializer='identity'),
            GroupNormalization(groups=1, axis=-1, epsilon=1e-12),
        ])

    def call(self, lfads_input, use_logrates=tf.constant(False)):
        """Passes data through low-dim readin and aligner
            overloads the call function from the base class, then calls it after
            setting the values of the encoder input

        Parameters
        ----------
        lfads_input : lfads_tf2.tuples.LFADSInput
            A namedtuple of tensors containing the data, external inputs, and encoder inputs.
        """
        data = lfads_input.enc_input
        if self.cfg.MODEL.NORMALIZE:
            data = self.norm_layer(data)
        lowd_data = self.lowd_readin(data)
        aligned_data = self.generator(lowd_data)
        lfads_input_out = LFADSInput(
            enc_input = aligned_data,
            ext_input = lfads_input.ext_input,
            dataset_name = lfads_input.dataset_name,
            behavior = lfads_input.behavior)
        # need to use grandparent call function here, so that lfads_input
        # is not overwritten by DimReducedLFADS call
        return super(DimReducedLFADS, self).call(lfads_input_out, use_logrates)

    def posterior_sample_and_average_call(self, input_data, n_samples):
        data = input_data.enc_input
        if self.cfg.MODEL.NORMALIZE:
            data = self.norm_layer(data)
        lowd_data = self.lowd_readin(data)
        aligned_data = self.generator(lowd_data)
        lfads_input = LFADSInput(
            enc_input = aligned_data,
            ext_input = input_data.ext_input,
            dataset_name = input_data.dataset_name,
            behavior = input_data.behavior)
        # need to use grandparent call function here, so that lfads_input
        # is not overwritten by DimReducedLFADS call
        return super(DimReducedLFADS, self).posterior_sample_and_average_call(lfads_input, n_samples)
