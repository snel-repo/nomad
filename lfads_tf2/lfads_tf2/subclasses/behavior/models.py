import os
import h5py
import pickle
import numpy as np
from os import path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow_addons.layers import GroupNormalization

from lfads_tf2.models import LFADS
from lfads_tf2.initializers import variance_scaling
from lfads_tf2.regularizers import DynamicL2
from lfads_tf2.tuples import LFADSInput, LFADSOutput

from lfads_tf2.subclasses.behavior.defaults import get_cfg_defaults

class BehaviorLFADS(LFADS):
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
        super(BehaviorLFADS, self).build_model_from_init_call(**kwargs)
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

        # Linear mapping from DECODE_FROM to behavioral variable 
        self.map2behavior = Sequential([
            Dense(
                self.cfg.MODEL.BEHAVIOR_DIM,
                kernel_initializer=variance_scaling,
                name='behavior_decoder')
        ])

        self.behavior_learning_rate = tf.Variable(self.cfg.TRAIN.LR.DECODER, trainable=False)
        self.behavior_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.behavior_learning_rate,
            epsilon=self.cfg.TRAIN.ADAM_EPSILON)

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
                
    def update_behavioral_learning_rate(self):
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
                winmax_val_loss = self.train_df.iloc[-(cfg.PATIENCE+1):-1].val_beh_mse.max()
                cur_val_loss = self.train_df.at[prev_epoch, 'val_beh_mse']
                # if the current val_loss is greater than the max in the window, decay LR
                if cur_val_loss > winmax_val_loss:
                    new_lr = max([cfg.DECAY * self.behavior_learning_rate.numpy(), cfg.STOP])
                    self.behavior_learning_rate.assign(new_lr)
        # report the current learning rate to the metrics
        new_lr = self.behavior_learning_rate.numpy()
        return new_lr

    def train_epoch(self, loadable_data=None):
        new_beh_lr = self.update_behavioral_learning_rate()
        return super(BehaviorLFADS, self).train_epoch(loadable_data=loadable_data)        

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
        lfads_output = super(BehaviorLFADS, self).call(
            lfads_input_lowd, use_logrates)

        predicted_behavior = self.map2behavior(getattr(lfads_output, self.cfg.TRAIN.DECODE.FROM))
        
        lfads_output = LFADSOutput(rates=lfads_output.rates,
                            ic_means=lfads_output.ic_means,
                            ic_stddevs=lfads_output.ic_stddevs,
                            ic_samps=lfads_output.ic_samps,
                            co_means=lfads_output.co_means,
                            co_stddevs=lfads_output.co_stddevs,
                            factors=lfads_output.factors,
                            gen_states=lfads_output.gen_states,
                            gen_init=lfads_output.gen_init,
                            gen_inputs=lfads_output.gen_inputs,
                            con_states=lfads_output.con_states,
                            predicted_behavior=predicted_behavior)

        return lfads_output

    def posterior_sample_and_average_call(self, lfads_input, n_samples):
        """Passes data through low-dim readin before performing posterior
        sampling.

        Overloads the function from the base class, then calls it after
        updating the value of the encoder input. Args and kwargs are 
        identical to the superclass.
        """
        data, ext_input, dataset_name, behavior = lfads_input
        if self.cfg.MODEL.NORMALIZE: 
            data = self.norm_layer(data)
        enc_input = self.lowd_readin(data)
        # Create low-D input for LFADS
        lfads_input_lowd = LFADSInput(
            enc_input=enc_input, ext_input=ext_input, dataset_name=dataset_name, behavior=behavior)
        # Perform posterior sampling based on low-dim input
        output, non_averaged_outputs = super(BehaviorLFADS, self) \
            .posterior_sample_and_average_call(lfads_input_lowd, n_samples)
        # output is decoder output tuple 
        # factors are third element, gen_states are fourth 
        if self.cfg.TRAIN.DECODE.FROM == 'factors': 
            predictor = output[2]
        elif self.cfg.TRAIN.DECODE.FROM == 'gen_states': 
            predictor = output[3]

        predicted_behavior = self.map2behavior(predictor)
        output[-1] = predicted_behavior

        return output, non_averaged_outputs

    def weighted_l2_loss(self):
        """ Calculate the L2 loss for the DimReducedLFADS model.

        This function computes the weighted L2 across all of the 
        recurrent kernels in LFADS, plus the readin kernel.
        """
        l2 = super(BehaviorLFADS, self).weighted_l2_loss()
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
        def compute_decoding_ramping_weight(start_epoch, increase_epoch):
            cur_epoch = self.cur_epoch
            if increase_epoch > 0:
                ratio = (cur_epoch - start_epoch) / (increase_epoch + 1)
                clipped_ratio = tf.clip_by_value(ratio, 0, 1)
                return tf.cast(clipped_ratio, tf.float32)
            else:
                return 1.0 if cur_epoch >= start_epoch else 0.0

        # ----- TRAINING STEP -----
        if self.training:
            with tf.GradientTape() as tape:
                
                variables_to_optimize = self.trainable_variables

                # perform the forward pass, using SV and / or CD as necessary
                nll_heldin, nll_heldout, posterior_params = self.train_call(batch)

                beh_ramping_weight = compute_decoding_ramping_weight(
                        self.cfg.TRAIN.DECODE.START_EPOCH, self.cfg.TRAIN.DECODE.INCREASE_EPOCH)
                beh_loss_scale = self.cfg.TRAIN.DECODE.SCALE

                predicted_behavior = self.call(self.batch_to_LFADSInput(batch)).predicted_behavior
                delay = self.cfg.TRAIN.DECODE.N_DELAY_BINS
                end_pt = self.cfg.MODEL.SEQ_LEN - delay
                beh_mse = tf.reduce_mean(tf.square(batch.behavior[:,delay:,:] - predicted_behavior[:,0:end_pt,:]))

                l2 = self.weighted_l2_loss()
                kl = self.weighted_kl_loss(*posterior_params)

                loss = nll_heldin + self.l2_ramping_weight * l2 \
                    + self.kl_ramping_weight * kl + beh_ramping_weight * beh_loss_scale * beh_mse
            
                scaled_loss = self.loss_scale * loss

            # compute gradients and descend
            lfads_inds = np.array([int(x) for x in range(len(variables_to_optimize)) if 'behavior' not in variables_to_optimize[x].name])
            behavior_inds = np.array([int(x) for x in range(len(variables_to_optimize)) if 'behavior' in variables_to_optimize[x].name])
            gradients = tape.gradient(scaled_loss, variables_to_optimize)
            gradients, gnorm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            
            gradients = np.array(gradients)
            variables_to_optimize = np.array(variables_to_optimize)
            
            self.optimizer.apply_gradients(zip(gradients[lfads_inds], variables_to_optimize[lfads_inds]))
            self.behavior_optimizer.apply_gradients(zip(gradients[behavior_inds], variables_to_optimize[behavior_inds]))

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
            # perform the forward pass, using SV and / or CD as necessary
            nll_heldin, nll_heldout, posterior_params = self.train_call(batch)

            beh_ramping_weight = compute_decoding_ramping_weight(
                    self.cfg.TRAIN.DECODE.START_EPOCH, self.cfg.TRAIN.DECODE.INCREASE_EPOCH)
            beh_loss_scale = self.cfg.TRAIN.DECODE.SCALE
            predicted_behavior = self.call(self.batch_to_LFADSInput(batch)).predicted_behavior
            delay = self.cfg.TRAIN.DECODE.N_DELAY_BINS
            end_pt = self.cfg.MODEL.SEQ_LEN - delay
            beh_mse = tf.reduce_mean(tf.square(batch.behavior[:,delay:,:] - predicted_behavior[:,0:end_pt,:]))

            l2 = self.weighted_l2_loss()
            kl = self.weighted_kl_loss(*posterior_params)

            loss = nll_heldin + self.l2_ramping_weight * l2 \
                + self.kl_ramping_weight * kl + beh_ramping_weight * beh_loss_scale * beh_mse

            if len(self.train_df) > 0:
                self.train_df['val_beh_mse'] = beh_mse.numpy()
            # record training statistics
            self._update_metrics({
                'val_loss': loss,
                'val_nll_heldin': nll_heldin,
                'val_nll_heldout': nll_heldout,
                'val_wt_kl': kl,
                'val_wt_l2': l2,
            }, batch_size=batch.data.shape[0])