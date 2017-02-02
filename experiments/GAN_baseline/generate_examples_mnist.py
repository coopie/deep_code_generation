"""
Generates random results from using the decoder
"""
from tqdm import trange
from scipy.misc import imsave

import numpy as np
import tensorflow as tf

from models import build_generator, gan_arg_scope

BASEDIR = 'experiments/GAN_baseline/'


def generate_examples():
    hidden_dim = 32
    z = tf.placeholder(tf.float32, shape=(1, 32), name='z')
    with tf.variable_scope('model'):
        with gan_arg_scope():
            generator_output = build_generator(z)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        experiment_checkpoint = '/mnist_digits_1'
        saver.restore(sess, tf.train.latest_checkpoint(BASEDIR + experiment_checkpoint))

        def g(latent_rep=None):
            if latent_rep is None:
                latent_rep = np.random.normal(0, 1, hidden_dim)
            return sess.run(
                        generator_output,
                        feed_dict={
                            'z:0': np.reshape(latent_rep, (1, hidden_dim))
                        }
                    )

        for i in trange(5):
            generated = g()[0].reshape((28, 28))

            imsave(BASEDIR + experiment_checkpoint + '_examples/{}.png'.format(i), generated.T)

            # tokens = np.argmax(generated, axis=-1)
            #
            # def token_to_string(t):
            #     if t == 0:
            #         return ''
            #     return TOKEN_MAP[t-1]
            #
            # text = ' '.join([token_to_string(t) for t in tokens])
            # with open(BASEDIR + 'simple_examples/{}.hs'.format(i), 'w') as f:
            #     f.write(text)


if __name__ == '__main__':
    with tf.Graph().as_default():
        generate_examples()
