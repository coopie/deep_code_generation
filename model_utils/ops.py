import tensorflow as tf


def vae_resampling(z_mus, z_log_sigmas, epsilon_std):
    def sampling(z_mean, z_log_sigma):
        epsilon = tf.random_normal(
            shape=tf.shape(z_mean),
            mean=0., stddev=epsilon_std
        )
        return z_mean + tf.exp(z_log_sigma) * epsilon

    z_resampled = tf.identity(sampling(z_mus, z_log_sigmas), name='decoder_input')
    return z_resampled
