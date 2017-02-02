"""
Code for building GANs. Inspired from
https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW/blob/master/utils.py and modified.
"""
import project_context  # NOQA

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import slim
from model_utils.loss_functions import discriminator_loss, generator_loss

# Names for the tensors representing input or output
NAMES = {
    'discriminator_input',
    'discriminator_output',
    'generator_input',
    'generator_output',
    'generator_loss',
    'discriminator_loss',
    'train_discriminator',
    'train_generator'
}


def gan_arg_scope():
    return slim.arg_scope(
        [layers.conv2d, layers.conv2d_transpose],
        activation_fn=tf.nn.elu,
        normalizer_fn=layers.batch_norm,
        normalizer_params={'scale': True}
    )


def build_mnist_gan_for_training(mnist_batch_queue):
    batch_size = mnist_batch_queue.get_shape()[0].value
    hidden_size = 32
    learning_rate = 1e-3
    with gan_arg_scope():
        with tf.variable_scope('model'):
            discriminator_for_dataset = build_discriminator(mnist_batch_queue)
            # num_discriminator_params = len(tf.trainable_variables())
            generator = build_generator(tf.random_normal([batch_size, hidden_size]))

        with tf.variable_scope('model', reuse=True):
            discriminator_for_g = build_discriminator(generator, reuse=True)

        d_loss = discriminator_loss(discriminator_for_dataset, discriminator_for_g)
        tf.summary.scalar('discriminator_loss', d_loss)

        g_loss = generator_loss(generator)
        tf.summary.scalar('generator_loss', g_loss)

        params = tf.trainable_variables()
        # discriminator_params = params[num_discriminator_params:]
        # generator_params = params[:num_discriminator_params]
        discriminator_params = [p for p in params if p.name.startswith('model/discriminator')]
        generator_params = [p for p in params if p.name.startswith('model/generator')]

        # might need to pass in update_ops=[]
        train_discriminator = slim.learning.create_train_op(
            d_loss,
            tf.train.AdamOptimizer(learning_rate / 100),
            variables_to_train=discriminator_params
        )
        train_generator = slim.learning.create_train_op(
            g_loss,
            tf.train.AdamOptimizer(learning_rate),
            variables_to_train=generator_params
        )

        return {
            'train_discriminator': train_discriminator.name,
            'train_generator': train_generator.name
        }


def build_discriminator(input_t, reuse=False):
    """Create encoder network.
    Args:
        input_tensor: a batch of images [batch_size, 1, 28, 28]
    Returns:
        A tensor that expresses the encoder network
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        net = layers.conv2d(input_t, 32, 5, stride=2)
        net = layers.conv2d(net, 64, 5, stride=2)
        net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
        net = layers.dropout(net, keep_prob=0.9)
        net = layers.flatten(net)
        return layers.fully_connected(net, 1, activation_fn=None)


def build_generator(input_tensor, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        net = tf.expand_dims(input_tensor, 1)
        net = tf.expand_dims(net, 1)
        net = layers.conv2d_transpose(net, 128, 3, padding='VALID')
        net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
        net = layers.conv2d_transpose(net, 32, 5, stride=2)
        net = layers.conv2d_transpose(
            net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
        # net = layers.flatten(net)
        return net
