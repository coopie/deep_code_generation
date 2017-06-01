"""
Makes examples of computed values of an autoencoder

* an autoencoded examples
* generated from a random latent vecor
* a markov chain generated vector

Usage:
    model_analysis.py [--basic] <option>
    model_analysis.py -h | --help

Options:
    -h --help   Show this screen.
    -b --basic  Use the basic huzzer dataset.

"""
from docopt import docopt
from sys import argv
from tqdm import trange
from scipy.misc import imsave
import os
import errno
import numpy as np
from random import randint
import project_context  # NOQA
from huzzer.tokenizing import TOKEN_MAP

from pipelines.data_sources import BASIC_DATASET_ARGS
from pipelines.one_hot_token import one_hot_token_dataset

import tensorflow as tf
from tensorflow.contrib import slim

from pipelines.data_sources import HuzzerSource, OneHotVecotorizer, TokenDatasource
from models import (  # NOQA
    build_conv1_encoder,
    build_decoder,
    build_conv1_decoder,
    build_conv1_encoder,
    conv_arg_scope,
    conv_arg_scope2,
    build_special_conv_encoder,
    build_special_conv_decoder,
    build_special_conv2_encoder,
    build_special_conv2_decoder,
    build_special_conv4_encoder,
    build_special_conv4_decoder
)


BASEDIR = 'experiments/VAE_baseline/'

NUMBER_OF_EXAMPLES = 10

def analyze_model(option, use_basic_dataset):
    # if option == 'simple':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_simple()
    # elif option == 'conv1':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_conv1()
    # elif option == 'conv2':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_conv2()
    # elif option == 'conv3':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_conv3()
    # elif option == 'conv4':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_conv4()
    # elif option == 'simple_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_simple_sss()
    # elif option == 'simple_double_latent_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_simple_sss(64)
    # elif option == 'simple_256_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_simple_sss(256)
    # elif option == 'simple_256_1k':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_simple_sss(256)
    # elif option == 'simple_1024_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_simple_sss(1024)
    # elif option == 'simple_1024_10k':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_simple_sss(1024)
    # elif option == 'simple_1024_1k':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_simple_sss(1024)
    # elif option == 'simple_1024_no_limit_1k':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_simple_sss(1024)
    # elif option == 'simple_8192_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_simple_sss(8192)
    # elif option == 'conv_special_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_special_conv()
    # elif option == 'conv_special_low_kl_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_special_conv()
    # elif option == 'conv_special2_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_special_conv2()
    # elif option == 'conv_special2_l1_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_special_conv2_l1()
    # elif option == 'conv_special3_l1_128_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_special_conv3_l1(128)
    # elif option == 'conv_special3_l1_256_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_special_conv3_l1(256)
    # elif option == 'conv_special3_big_l1_512_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_special_conv3_l1(
    #         512, filter_length=10
    #     )
    # elif option == 'conv_special4_l1_1024_sss':
    #     data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output = make_special_conv4_l1(
    #         1024, filter_length=3, num_filters=256
    #     )
    # else:
    sequence_cap = 56 if use_basic_dataset else 130

    if option.startswith('simple'):
        z_size = int(option.split('_')[1])
        _, encoder_input, encoder_output, decoder_input, decoder_output = make_simple(z_size, sequence_cap)
    else:
        print('INVALID OPTION {}'.format(option))
        exit()

    model_directory = ('basic_' if use_basic_dataset else '') + option

    huzzer_kwargs = BASIC_DATASET_ARGS if use_basic_dataset else {}
    sequence_cap = 56 if use_basic_dataset else 130

    print('Setting up data pipeline...')
    dataset = one_hot_token_dataset(
        batch_size=1,
        number_of_batches=1000,
        cache_path='{}model_analysis_simple'.format(
            'basic_' if use_basic_dataset else ''
        ),
        length=sequence_cap,
        huzzer_kwargs=huzzer_kwargs
    )

    saver = tf.train.Saver()
    with tf.Session() as sess:

        restore_dir = tf.train.latest_checkpoint(BASEDIR + '{}'.format(model_directory))
        print('resoring sesstion at : ' + restore_dir)
        saver.restore(
            sess, restore_dir
        )

        def g(latent_rep=None, variance=1):
            if latent_rep is None:
                latent_rep = np.random.normal(0, variance, decoder_input.get_shape()[-1].value)
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
        latent_sampling_dir = examples_dir + '/generated/'
        autoencoded_dir = examples_dir + '/autoencoded/'
        # markov_generating_dir = examples_dir + '/markov_generating/'

        mkdir_p(examples_dir)
        mkdir_p(latent_sampling_dir)
        mkdir_p(autoencoded_dir)
        mkdir_p(markov_generating_dir)

        #  Autoencode bit

        for i in trange(NUMBER_OF_EXAMPLES):
            dir_for_example = os.path.join(autoencoded_dir, str(i))
            mkdir_p(dir_for_example)
            example_input = data_pipeline[str(i)].astype('float32')
            input_text = example_to_code(example_input)

            latent_rep, _ = e(example_input)

            latent_image = latent_rep.reshape((latent_rep.size // 32, 32))

            imsave(
                dir_for_example + '/{}_latent.png'.format(i),
                latent_image
            )
            reconstrcted_tokens, _ = g(latent_rep)
            autoencoded_text = example_to_code(reconstrcted_tokens)
            imsave(dir_for_example + '/input.png', example_input.T)
            imsave(dir_for_example + '/decoder_output.png', reconstrcted_tokens.T)

            with open(dir_for_example + '/input.hs', 'w') as f:
                f.write(input_text)

            with open(dir_for_example + '/autoencoded_code.hs', 'w') as f:
                f.write(autoencoded_text)


def example_to_code(example):
    tokens = np.argmax(example, axis=-1)
    text = ' '.join([token_to_string(t) for t in tokens])
    return text


def token_to_string(t):
    if t == 0:
        return ''
    return TOKEN_MAP[t]


def make_simple(latent_dim, sequence_length):
    huzz = HuzzerSource()
    data_pipeline = OneHotVecotorizer(
        TokenDatasource(huzz),
        54,
        256
    )

    x_shape = (sequence_length, 54)
    encoder_input = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
    x_flat = slim.flatten(encoder_input)
    z = slim.fully_connected(
        x_flat, latent_dim, scope='encoder_output', activation_fn=tf.tanh
    )
    encoder_output = tf.identity(z, 'this_is_output')

    decoder_input = tf.placeholder(tf.float32, shape=(1, latent_dim), name='decoder_input')
    decoder_output = build_decoder(decoder_input, x_shape)
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


def make_conv4():
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


def make_simple_sss(latent_dim=32):
    x_shape = (128, 54)
    huzz = HuzzerSource()
    data_pipeline = OneHotVecotorizer(
        TokenDatasource(huzz),
        x_shape[1],
        x_shape[0]
    )

    encoder_input = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
    x_flat = slim.flatten(encoder_input)
    z = slim.fully_connected(
        x_flat, latent_dim, scope='encoder_output', activation_fn=tf.tanh
    )

    decoder_input = tf.placeholder(tf.float32, shape=(1, latent_dim), name='decoder_input')
    decoder_output = build_decoder(
        decoder_input, x_shape, activation=tf.nn.relu6
    )
    decoder_output = tf.reshape(decoder_output, x_shape)
    return data_pipeline, encoder_input, z, decoder_input, decoder_output


def make_special_conv():
    latent_dim = 64
    x_shape = (128, 54)
    huzz = HuzzerSource()
    data_pipeline = OneHotVecotorizer(
        TokenDatasource(huzz),
        x_shape[1],
        x_shape[0]
    )

    decoder_input = tf.placeholder(tf.float32, shape=(1, latent_dim), name='decoder_input')
    encoder_input = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
    with conv_arg_scope():
        encoder_output, _ = build_special_conv_encoder(encoder_input, latent_dim)
        decoder_output = build_special_conv_decoder(decoder_input, x_shape)
        decoder_output = tf.nn.softmax(decoder_output, dim=-1)
        decoder_output = tf.squeeze(decoder_output, 0)

    return data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output


def make_special_conv2():
        latent_dim = 64
        x_shape = (128, 54)
        huzz = HuzzerSource()
        data_pipeline = OneHotVecotorizer(
            TokenDatasource(huzz),
            x_shape[1],
            x_shape[0]
        )

        num_filters = 64
        decoder_input = tf.placeholder(tf.float32, shape=(1, latent_dim), name='decoder_input')
        encoder_input = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
        with conv_arg_scope():
            encoder_output, _ = build_special_conv2_encoder(encoder_input, latent_dim, num_filters)
            decoder_output = build_special_conv2_decoder(decoder_input, x_shape, num_filters)
            decoder_output = tf.nn.softmax(decoder_output, dim=-1)
            decoder_output = tf.squeeze(decoder_output, 0)

        return data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output


def make_special_conv2_l1():
        latent_dim = 64
        x_shape = (128, 54)
        huzz = HuzzerSource()
        data_pipeline = OneHotVecotorizer(
            TokenDatasource(huzz),
            x_shape[1],
            x_shape[0]
        )

        num_filters = 64
        decoder_input = tf.placeholder(tf.float32, shape=(1, latent_dim), name='decoder_input')
        encoder_input = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
        with conv_arg_scope2():
            encoder_output, _ = build_special_conv2_encoder(encoder_input, latent_dim, num_filters)
            decoder_output = build_special_conv2_decoder(decoder_input, x_shape, num_filters)
            decoder_output = tf.nn.softmax(decoder_output, dim=-1)
            decoder_output = tf.squeeze(decoder_output, 0)

        return data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output


def make_special_conv3_l1(latent_dim, filter_length=5):
    x_shape = (128, 54)
    huzz = HuzzerSource()
    data_pipeline = OneHotVecotorizer(
        TokenDatasource(huzz),
        x_shape[1],
        x_shape[0]
    )

    num_filters = 64
    decoder_input = tf.placeholder(tf.float32, shape=(1, latent_dim), name='decoder_input')
    encoder_input = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
    with conv_arg_scope2():
        encoder_output, _ = build_special_conv2_encoder(
            encoder_input, latent_dim, num_filters, filter_length=filter_length
        )
        decoder_output = build_special_conv2_decoder(
            decoder_input, x_shape, num_filters, filter_length=filter_length
        )
        decoder_output = tf.nn.softmax(decoder_output, dim=-1)
        decoder_output = tf.squeeze(decoder_output, 0)

    return data_pipeline, encoder_input, encoder_output, decoder_input, decoder_output


def make_special_conv4_l1(latent_dim, filter_length=3, num_filters=256):
        x_shape = (128, 54)
        huzz = HuzzerSource()
        data_pipeline = OneHotVecotorizer(
            TokenDatasource(huzz),
            x_shape[1],
            x_shape[0]
        )

        decoder_input = tf.placeholder(tf.float32, shape=(1, latent_dim), name='decoder_input')
        encoder_input = tf.placeholder(tf.float32, shape=(1, *x_shape), name='encoder_input')
        with conv_arg_scope2():
            encoder_output, _, dense_layer_size = build_special_conv4_encoder(
                encoder_input, latent_dim, num_filters, filter_length=filter_length
            )
            decoder_output = build_special_conv4_decoder(
                decoder_input, x_shape, num_filters, filter_length=filter_length, dense_layer_size=dense_layer_size
            )
            decoder_output = tf.nn.softmax(decoder_output, dim=-1)
            decoder_output = tf.squeeze(decoder_output, 0)

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
    args = docopt(__doc__, version='N/A')
    option = args.get('<option>')
    use_basic_dataset = args.get('--basic')
    analyze_model(option, use_basic_dataset)
