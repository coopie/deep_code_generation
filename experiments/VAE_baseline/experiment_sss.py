"""
experiments similar to those in experiment.py, but with limited dataset
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
    NUMBER_BATCHES = 500
    dataset = one_hot_token_dataset(
        BATCH_SIZE,
        NUMBER_BATCHES,
        length=128
    )

    def get_batch():
        return dataset()[0]


    # use the queue for training
    queue = build_single_output_queue(get_batch, (BATCH_SIZE, *X_SHAPE))
    x = queue.dequeue(name='encoder_input')
    if option == 'simple':
        tensor_names = build_simple_network2(x, X_SHAPE, 32)
    if option == 'simple_double_latent':
        tensor_names = build_simple_network2(x, X_SHAPE, 64)
    if option == 'conv_special':
        tensor_names = build_special_conv(x, X_SHAPE, 64)
    else:
        print('INVALID OPTION')
        exit(1)

    logdir = 'experiments/VAE_baseline/{}_sss'.format(option)

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
