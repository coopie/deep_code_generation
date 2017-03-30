"""
experiments similar to those in experiment.py, but with limited dataset of approx. 10,000 examples
"""


import project_context  #  NOQA

from sys import argv
import numpy as np

import logging
from pipelines.one_hot_token import one_hot_token_dataset
from model_utils.queues import build_single_output_queue

from models import (
    build_simple_network2,
    build_special_conv,
    build_special_conv_low_kl,
    build_special_conv2,
    build_special_conv2_l1,
    build_special_conv4_l1
)

import tensorflow as tf
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training.supervisor import Supervisor
from tensorflow.python.summary import summary
tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim


def run_experiment(option):
    BATCH_SIZE = 128
    X_SHAPE = (128, 54)

    # set up pipeline
    print('Setting up data pipeline')
    NUMBER_BATCHES = 80
    dataset = one_hot_token_dataset(
        BATCH_SIZE,
        NUMBER_BATCHES,
        length=128,
        cache_path='one_hot_token_haskell_batch{}_number{}'.format(BATCH_SIZE, NUMBER_BATCHES)
    )

    def get_batch():
        return dataset()[0]


    # use the queue for training
    queue = build_single_output_queue(get_batch, (BATCH_SIZE, *X_SHAPE))
    x = queue.dequeue(name='encoder_input')
    if option == 'simple':
        tensor_names = build_simple_network2(x, X_SHAPE, 32)
    elif option == 'simple_double_latent':
        tensor_names = build_simple_network2(x, X_SHAPE, 64)
    elif option == 'simple_256':
        tensor_names = build_simple_network2(x, X_SHAPE, 256)
    elif option == 'simple_1024':
        tensor_names = build_simple_network2(x, X_SHAPE, 1024)
    elif option == 'simple_8192':
        tensor_names = build_simple_network2(x, X_SHAPE, 8192)
    elif option == 'conv_special':
        tensor_names = build_special_conv(x, X_SHAPE, 64)
    elif option == 'conv_special_low_kl':
        tensor_names = build_special_conv_low_kl(x, X_SHAPE, 64)
    elif option == 'conv_special2':
        tensor_names = build_special_conv2(x, X_SHAPE, 64)
    elif option == 'conv_special2_l1':
        tensor_names = build_special_conv2_l1(x, X_SHAPE, 64)
    elif option == 'conv_special2_l1_128':
        tensor_names = build_special_conv2_l1(x, X_SHAPE, 128)

    # conv3 is conv2 but with initial filter length of 5 instead of 1
    elif option == 'conv_special3_l1_128':
        tensor_names = build_special_conv2_l1(x, X_SHAPE, 128, filter_length=5)
    elif option == 'conv_special3_l1_256':
        tensor_names = build_special_conv2_l1(x, X_SHAPE, 256, filter_length=5)
    elif option == 'conv_special3_l1_128f_256':
        tensor_names = build_special_conv2_l1(x, X_SHAPE, 256, filter_length=5, num_filters=128)
    elif option == 'conv_special3_big_l1_512':
        tensor_names = build_special_conv2_l1(x, X_SHAPE, 512, filter_length=10)
    elif option == 'conv_special4_l1_1024':
        tensor_names = build_special_conv4_l1(x, X_SHAPE, 1024, filter_length=3, num_filters=256)
    elif option == 'conv_special4_l1_2048_f5':
        tensor_names = build_special_conv4_l1(x, X_SHAPE, 1024, filter_length=5, num_filters=256)
    else:
        print('INVALID OPTION')
        exit(1)

    logdir = 'experiments/VAE_baseline/{}_10k'.format(option)

    sv = Supervisor(
        logdir=logdir,
        save_summaries_secs=20,
        save_model_secs=120
    )
    # Get a TensorFlow session managed by the supervisor.
    with sv.managed_session() as sess:
        # Use the session to train the graph.
        while not sv.should_stop():
            sess.run(
                'train_on_batch',
            )


def make_example_uri(ident):
    return '{0}'.format(ident)

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO
    )

    args = argv[1:]
    assert len(args) == 1, 'You must provide one argument'
    option = args[0]

    with tf.Graph().as_default():
        run_experiment(option)
