import tensorflow as tf
from tensorflow.contrib import losses


def discriminator_loss(discriminator_real, discriminator_generated):
    return (
        losses.sigmoid_cross_entropy(
            discriminator_real, tf.ones(tf.shape(discriminator_real))
        ) +
        losses.sigmoid_cross_entropy(
            discriminator_generated, tf.zeros(tf.shape(discriminator_real))
        )
    )


def generator_loss(discriminator_generated):
    return losses.sigmoid_cross_entropy(
        discriminator_generated,
        tf.ones(tf.shape(discriminator_generated))
    )
