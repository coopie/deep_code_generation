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


def vae_cross_entropy_loss(
    x,
    x_decoded,
    mus,
    log_sigmas,
    kl_limit=None,
    kl_scale=None
):
    """
    Similar to vae_loss, but uses cross-entropy rather than simple reconstruction_loss

    NOTE x_decoded must be in logits, not softmax!
    """

    # get labels from x
    labels = tf.argmax(x, 2)  # HACK!

    cross_entropy_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=x_decoded
        )
    )

    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

    kl_loss = - 0.5 * tf.reduce_mean(
        1 + log_sigmas - tf.square(mus) - tf.exp(log_sigmas)
    )
    if kl_limit is not None:
        kl_loss = tf.maximum(kl_loss, tf.constant(kl_limit))
    if kl_scale is not None:
        kl_loss = kl_scale * kl_loss
    tf.summary.scalar('kl_loss', kl_loss)

    total_loss = tf.add(cross_entropy_loss, kl_loss, name='loss')
    tf.summary.scalar('total_loss', total_loss)
    return total_loss


# For TF Fold
def softmax_crossentropy(logit, label_vector):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels=label_vector,
        logits=logit
    )


def kl_divergence(mus_and_log_sigs):

    halfway = int(mus_and_log_sigs.get_shape()[1].value / 2)  # HACK: make this cleaner
    mus = mus_and_log_sigs[:, :halfway]
    log_sigs = mus_and_log_sigs[:, halfway:]

    kl_loss_term = -0.5 * tf.reduce_mean(
        1 + log_sigs - tf.square(mus) - tf.exp(log_sigs),
        axis=1
    )

    return kl_loss_term
