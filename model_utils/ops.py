import tensorflow as tf


def vae_resampling(z_mus, z_log_sigmas, epsilon_std):
    def sampling(z_mean, z_log_sigma):
        epsilon = tf.random_normal(
            shape=tf.shape(z_mean),
            mean=0., stddev=epsilon_std
        )
        return z_mean + tf.exp(z_log_sigma) * epsilon

    # for legacy reasons
    z_resampled = tf.identity(sampling(z_mus, z_log_sigmas), name='decoder_input')
    return z_resampled


def resampling(mus_and_log_sigs):
    """
    Batch resampling operation, mus_and_log_sigs of shape (b x z_size*2)
    """
    z_size = mus_and_log_sigs.get_shape()[-1].value // 2
    mus = mus_and_log_sigs[:, :z_size]
    log_sigs = mus_and_log_sigs[:, z_size:]

    z_resampled = vae_resampling(mus, log_sigs, epsilon_std=0.01)
    return z_resampled


def get_sequence_lengths(x):
    is_padding = x[:, :, 0]
    return (x.get_shape()[1].value - tf.reduce_sum(is_padding, axis=1)) + 1


def compute_attention_vector(previous_hidden_states, unnormalized_attention_coefs):
    attention_coefs = tf.nn.softmax(
        unnormalized_attention_coefs, dim=-1
    )
    return tf.reduce_sum(
        tf.multiply(
            previous_hidden_states,
            tf.expand_dims(attention_coefs, -1)
        ),
        axis=1
    ), attention_coefs
