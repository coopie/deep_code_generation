"""
Makes examples of computed values of an autoencoder

* an autoencoded examples
* generated from a random latent vecor
* a markov chain generated vector
"""
import project_context  # NOQA
from sys import argv
from tqdm import trange
from scipy.misc import imsave
import os
import errno
import numpy as np
from random import randint
from huzzer.tokenizing import TOKEN_MAP

import tensorflow as tf
from tensorflow.contrib import slim

from pipelines.data_sources import HuzzerSource, OneHotVecotorizer, TokenDatasource
from models import (  # NOQA
    build_conv1_encoder, build_decoder, build_conv1_decoder, build_conv1_encoder, conv_arg_scope
)


BASEDIR = 'experiments/VAE_baseline/'


def generate_code(option):
    if option == 'simple':
        data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_simple()
    elif option == 'conv1':
        data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_conv1()
    elif option == 'conv2':
        data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_conv2()
    elif option == 'conv3':
        data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_conv3()
    else:
        print('INVALID OPTION')
        exit()

    import code  # NOQA
    code.interact(local=locals())

    saver = tf.train.Saver()
    with tf.Session() as sess:

        saver.restore(
            sess, tf.train.latest_checkpoint(BASEDIR + '{}'.format(option))
        )

        def g(latent_rep=None):
            if latent_rep is None:
                latent_rep = np.random.normal(0, 1, decoder_input.get_shape()[-1].value)
            generated = sess.run(
                decoder_output,
                feed_dict={
                    decoder_input: np.reshape(latent_rep, (1, -1))
                }
            )
            return generated, latent_rep

        def e(example_data=None):
            if example_data is None:
                key = str(randint(0, 100000000))
                example_data = data_pipeline[key]
            example_data = np.reshape(example_data, (1, *example_data.shape))
            encoded = sess.run(
                encoder_output,
                feed_dict={
                    encoder_input: example_data
                }
            )
            return encoded, example_data

        examples_dir = BASEDIR + '{}_examples'.format(option)
        latent_sampling_dir = examples_dir + '/latent_sampling/'
        autoencoded_dir = examples_dir + '/autoencoded/'
        markov_generating_dir = examples_dir + '/markov_generating/'

        mkdir_p(examples_dir)
        mkdir_p(latent_sampling_dir)
        mkdir_p(autoencoded_dir)
        mkdir_p(markov_generating_dir)

        for i in trange(5):
            example_input = data_pipeline[str(i)].astype('float32')
            input_text = example_to_code(example_input)

            latent_rep, _ = e(example_input)

            reconstrcted_tokens, _ = g(latent_rep)
            autoencoded_text = example_to_code(reconstrcted_tokens)
            imsave(autoencoded_dir + '{}_input.png'.format(i), example_input.T)
            imsave(autoencoded_dir + '{}_output.png'.format(i), reconstrcted_tokens.T)

            with open(autoencoded_dir + '{}_input.hs'.format(i), 'w') as f:
                f.write(input_text)

            with open(autoencoded_dir + '{}_output.hs'.format(i), 'w') as f:
                f.write(autoencoded_text)


def example_to_code(example):
    tokens = np.argmax(example, axis=-1)
    text = ' '.join([token_to_string(t) for t in tokens])
    return text


def token_to_string(t):
    if t == 0:
        return ''
    return TOKEN_MAP[t]


def make_simple():
    huzz = HuzzerSource()
    data_pipeline = OneHotVecotorizer(
        TokenDatasource(huzz),
        54,
        256
    )

    x_shape = (256, 54)
    latent_dim = 16
    encoder_input = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
    x_flat = slim.flatten(encoder_input)
    z = slim.fully_connected(
        x_flat, latent_dim, scope='encoder_output', activation_fn=tf.tanh
    )
    encoder_output = tf.identity(z, 'this_is_output')

    decoder_input = tf.placeholder(tf.float32, shape=(1, 16), name='decoder_input')
    decoder_output = build_decoder(decoder_input, (256, 54))['decoder_input']
    latent_dim = 16
    return data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output


def make_conv1():
    latent_dim = 16
    x_shape = (128, 54)
    huzz = HuzzerSource()
    data_pipeline = OneHotVecotorizer(
        TokenDatasource(huzz),
        x_shape[1],
        x_shape[0]
    )

    decoder_input = tf.placeholder(tf.float32, shape=(1, latent_dim), name='decoder_input')
    with conv_arg_scope():
        encoder_input = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
        encoder_output, _ = build_conv1_encoder(encoder_input, latent_dim)

        decoder_output = build_conv1_decoder(decoder_input, x_shape)
        decoder_output = tf.reshape(decoder_output, x_shape)

    return data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output


def make_conv2():
    latent_dim = 32
    x_shape = (128, 54)
    huzz = HuzzerSource()
    data_pipeline = OneHotVecotorizer(
        TokenDatasource(huzz),
        x_shape[1],
        x_shape[0]
    )

    decoder_input = tf.placeholder(tf.float32, shape=(1, latent_dim), name='decoder_input')
    with conv_arg_scope():
        encoder_input = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
        encoder_output, _ = build_conv1_encoder(encoder_input, latent_dim)

        decoder_output = build_conv1_decoder(decoder_input, x_shape)
        decoder_output = tf.nn.softmax(decoder_output, dim=-1)
        decoder_output = tf.reshape(decoder_output, x_shape)

    return data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output


def make_conv3():
    latent_dim = 64
    x_shape = (128, 54)
    huzz = HuzzerSource()
    data_pipeline = OneHotVecotorizer(
        TokenDatasource(huzz),
        x_shape[1],
        x_shape[0]
    )

    decoder_input = tf.placeholder(tf.float32, shape=(1, latent_dim), name='decoder_input')
    with conv_arg_scope():
        encoder_input = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
        encoder_output, _ = build_conv1_encoder(encoder_input, latent_dim)

        decoder_output = build_conv1_decoder(decoder_input, x_shape)
        decoder_output = tf.nn.softmax(decoder_output, dim=-1)
        decoder_output = tf.reshape(decoder_output, x_shape)

    return data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output


# echoes the behaviour of mkdir -p
# from http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == '__main__':
    args = argv[1:]
    assert len(args) == 1, 'you must give a experiment type'
    option = args[0]

    with tf.Graph().as_default():
        generate_code(option)
