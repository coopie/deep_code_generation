import project_context  # NOQA
import numpy as np
from random import randint
from huzzer.tokenizing import TOKEN_MAP

import tensorflow as tf

from models import build_decoder

from pipelines.data_sources import HuzzerSource, OneHotVecotorizer, TokenDataSource


slim = tf.contrib.slim

NAMES = {
    'encoder_input': 'encoder_input:0',
    'encoder_output': 'encoder_output:0',
    'decoder_input': 'decoder_input:0',
    'decoder_output': 'decoder_output:0',
    'loss': 'loss:0',
    'train_on_batch': 'train_on_batch:0'
}


def autoencode():
    huzz = HuzzerSource()

    data_pipeline = OneHotVecotorizer(
        TokenDataSource(huzz),
        54,
        256
    )


    x_shape = (256, 54)
    latent_dim = 16
    x = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
    x_flat = slim.flatten(x)
    z = slim.fully_connected(
        x_flat, latent_dim, scope='encoder_output', activation_fn=tf.tanh
    )
    z = tf.identity(z, 'this_is_output')

    z = tf.placeholder(tf.float32, shape=(1, 16), name='decoder_input')
    build_decoder(z, (256, 54))

    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('experiments/VAE_baseline/simple'))

        def a(example_data=None):
            if example_data is None:
                key = str(randint(0, 100000000))
                code = huzz[key]
                print('INPUT CODE FOR NETWORK:')
                print(code)
                example_data = data_pipeline[key]
            example_data = np.reshape(example_data, (1, *example_data.shape))
            return sess.run(
                ['this_is_output:0'],
                feed_dict={
                    NAMES['encoder_input']: example_data
                }
            )

        def g(latent_rep):
            return sess.run(
                NAMES['decoder_output'],
                feed_dict={
                    NAMES['decoder_input']: latent_rep[0]
                }
            )

        latent_reps = a()
        print('LATENT REPRESENTATION:')
        print(latent_reps)
        recon = g(latent_reps)[0]

        tokens = np.argmax(recon, axis=-1)

        def token_to_string(t):
            if t == 0:
                return ''
            return TOKEN_MAP[t-1]

        text = ' '.join([token_to_string(t) for t in tokens])
        print('REGENERATED CODE:')
        print(text)




if __name__ == '__main__':
    with tf.Graph().as_default():
        autoencode()
