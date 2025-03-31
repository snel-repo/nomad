import os
import h5py
import pickle
import numpy as np
from os import path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow_addons.layers import GroupNormalization

from lfads_tf2.models import LFADS
from lfads_tf2.initializers import variance_scaling
from lfads_tf2.regularizers import DynamicL2
from lfads_tf2.tuples import LFADSInput

from lfads_tf2.subclasses.dimreduced.defaults import get_cfg_defaults

class DimReducedLFADS(LFADS):
    """Modified LFADS model with low-D read-in network."""
    def get_cfg_defaults(self):
        """Returns the DimReducedLFADS configuration defaults.

        Returns
        -------
        yacs.config.CfgNode
            The default configuration for the model.
        """
        return get_cfg_defaults()

    def build_model_from_init_call(self, **kwargs):
        """Builds DimReduced LFADS model based on a configuration

        The superclass does most of the leg work, but this function adds
        a network that can optionally be initialized using factor analysis.
        Keyword args are identical to the superclass. Make sure to use 
        the defaults config from the subclass folder.
        """
        super(DimReducedLFADS, self).build_model_from_init_call(**kwargs)
        # Add linear mapping followed by layer normalization
        self.lowd_readin = Sequential([
            Dense(
                self.cfg.MODEL.ENC_INPUT_DIM,
                kernel_initializer=variance_scaling,
                kernel_regularizer=DynamicL2(
                scale=self.cfg.TRAIN.L2.READIN_SCALE),
                name='lowd_linear'),
            GroupNormalization(groups=1, axis=-1, epsilon=1e-12)
        ])

        if self.cfg.MODEL.NORMALIZE: 
            self.norm_layer = Sequential([
                Dense(
                self.cfg.MODEL.DATA_DIM,
                kernel_initializer=variance_scaling,
                kernel_regularizer=DynamicL2(
                scale=self.cfg.TRAIN.L2.READIN_SCALE),
                name='norm_layer'),
            ])

            norm_param_path = path.join(self.cfg.TRAIN.DATA.DIR, 'normalization.h5')
            f = h5py.File(norm_param_path, 'r')
            bias = f['bias'][()]
            mat = f['matrix'][()]
            f.close()

            if not self.from_existing:
                self.load_matrix_to_lowd_readin(mat, bias=bias, freeze=True, norm_layer=True)

        self.is_dayk = False

    def call(self, lfads_input, use_logrates=tf.constant(False)):
        """Passes data through low-dim readin, followed by LFADS encoders.

        Overloads the function from the base class, then calls it after
        updating the value of the encoder input. Args and kwargs are 
        identical to the superclass.
        """
        # Unpack the input and compute the encoder input
        data, ext_input, dataset_name, behavior = lfads_input
        if self.cfg.MODEL.NORMALIZE: 
            data = self.norm_layer(data)
        enc_input = self.lowd_readin(data)
        # Create low-D input for LFADS
        lfads_input_lowd = LFADSInput(
            enc_input=enc_input, ext_input=ext_input, dataset_name=dataset_name, behavior=behavior)
        # Pass data through LFADS
        lfads_output = super(DimReducedLFADS, self).call(
            lfads_input_lowd, use_logrates)
            
        return lfads_output

    def posterior_sample_and_average_call(self, lfads_input, n_samples):
        """Passes data through low-dim readin before performing posterior
        sampling.

        Overloads the function from the base class, then calls it after
        updating the value of the encoder input. Args and kwargs are 
        identical to the superclass.
        """
        # Unpack the input and compute the encoder input
        data, ext_input, dataset_name, behavior = lfads_input
        if self.cfg.MODEL.NORMALIZE: 
            data = self.norm_layer(data)
        enc_input = self.lowd_readin(data)
        # Create low-D input for LFADS
        lfads_input_lowd = LFADSInput(
            enc_input=enc_input, ext_input=ext_input, dataset_name=dataset_name, behavior=behavior)
        # Perform posterior sampling based on low-dim input
        output, non_averaged_outputs = super(DimReducedLFADS, self) \
            .posterior_sample_and_average_call(lfads_input_lowd, n_samples)

        return output, non_averaged_outputs

    def weighted_l2_loss(self):
        """ Calculate the L2 loss for the DimReducedLFADS model.

        This function computes the weighted L2 across all of the 
        recurrent kernels in LFADS, plus the readin kernel.
        """
        l2 = super(DimReducedLFADS, self).weighted_l2_loss()
        # Add the L2 cost of the readin, averaged over elements
        kernel = self.lowd_readin.get_layer('lowd_linear').kernel
        readin_size = tf.size(kernel, out_type=tf.float32)
        l2 += tf.reduce_sum(self.lowd_readin.losses) / readin_size
        return l2

    def load_matrix_to_lowd_readin(self, matrix, bias=None, freeze=False, norm_layer=False): 
        '''
        Assigns an input matrix to the weights of the lowD readin
        
        matrix: numpy.array
            array containing values to assign as lowD readin 
            weights
        freeze: bool
            whether to freeze these weights for training, 
            by default False
        '''
        data_shape, _, ext_input_shape, name_shape, beh_shape = self.get_input_shapes(10)
        noise = LFADSInput(
            enc_input=np.ones(shape=data_shape, dtype=np.float32),
            ext_input=np.ones(shape=ext_input_shape, dtype=np.float32),
            dataset_name=np.full(shape=name_shape, fill_value=''),
            behavior=np.ones(shape=beh_shape, dtype=np.float32)
        )
        self.call(noise) # initialize weights so they can be overwritten 

        if norm_layer: 
            self.norm_layer.build(data_shape)
            # assign weight matrix to essential kernel
            self.norm_layer.weights[0].assign(matrix)

            # assign bias term (will always be one)
            # we want to subtract the bias so multiply by negative 1
            self.norm_layer.weights[1].assign(-1*bias)

            # freeze all values 
            if freeze: 
                for v in self.norm_layer.trainable_variables: 
                    v._trainable = False

        else: # normal lowD readin matrix 
            # assign the sequential kernel to hold the matrix as weights 
            self.lowd_readin.weights[0].assign(matrix)

            # if a bias term is provided, assign this here 
            if bias is not None: 
                self.lowd_readin.weights[1].assign(bias)
            
            # freeze all assigned values values 
            if freeze: 
                self.lowd_readin.trainable_variables[0]._trainable = False
                
                if bias is not None: 
                    self.lowd_readin.trainable_variables[1]._trainable = False

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

        if not self.is_dayk: 
            super(DimReducedLFADS, self)._step(batch)
        else:
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
                gradients = tape.gradient(scaled_loss, self.trainable_vars)
                gradients, gnorm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_vars))

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