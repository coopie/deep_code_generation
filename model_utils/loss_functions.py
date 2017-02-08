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


def vae_loss(x, x_decoded, mus, log_sigmas, kl_limit=None):
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_decoded))
    tf.summary.scalar('reconstruction_loss', reconstruction_loss)

    kl_loss = - 0.5 * tf.reduce_mean(
        1 + log_sigmas - tf.square(mus) - tf.exp(log_sigmas)
    )
    if kl_limit is not None:
        kl_loss = tf.maximum(kl_loss, tf.constant(kl_limit))
    tf.summary.scalar('kl_loss', kl_loss)

    total_loss = tf.add(reconstruction_loss, kl_loss, name='loss')
    tf.summary.scalar('total_loss', total_loss)
    return total_loss
