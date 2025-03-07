import tensorflow as tf

class MinibatchDiscrimination(tf.keras.layers.Layer):
    """ Implements minibatch discrimination 
    
    input shape is batch_size x feature_dim
    output shape is batch_size x (feature_dim + num_kernels)
    reference: https://github.com/matteson/tensorflow-minibatch-discriminator/blob/master/discriminator.py
    """
    def __init__(self, num_kernels, kernel_dim):
        super(MinibatchDiscrimination, self).__init__()
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim

    def build(self, input_shape):
        with tf.name_scope('minibatch_disc'):
            num_input_features = int(input_shape[-1])
            self.kernel = self.add_weight(
                name='kernel',
                shape=[num_input_features, self.num_kernels, self.kernel_dim],
                initializer=tf.initializers.TruncatedNormal(),
                trainable=True,
            )

    def call(self, input):
        kernel_features = tf.einsum('ij,jkl->ikl', input, self.kernel)
        diffs = tf.map_fn(lambda x: x - kernel_features, tf.expand_dims(kernel_features, 1))
        minibatch_features = tf.reduce_sum(tf.exp(-tf.reduce_sum(tf.abs(diffs), axis=3)), axis=1)
        return tf.concat([input, minibatch_features], 1)