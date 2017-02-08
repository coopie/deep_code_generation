import project_context  #  NOQA

from sys import argv
import numpy as np

import logging
from pipelines.one_hot_token import one_hot_token_pipeline

from models import build_simple_network, build_queue, build_conv1, build_conv2

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
    data_pipeline = one_hot_token_pipeline(
        for_cnn=False,
        length=128
    )

    # Function to pass into queue
    batch_index = 0
    def get_batch():
        nonlocal batch_index

        if batch_index % 100 == 0:
            logging.info('{} examples used'.format(batch_index * BATCH_SIZE))
        code_seeds = [
            str(i) for i in range(
                batch_index * BATCH_SIZE,
                (batch_index + 1) * BATCH_SIZE
            )
        ]

        batch = np.array(data_pipeline[code_seeds])
        batch_index += 1
        return batch

    # use the queue for trai n ing
    queue = build_queue(get_batch, BATCH_SIZE, X_SHAPE)
    x = queue.dequeue(name='encoder_input')
    if option == 'simple':
        print('this no longer works: - a small refactor would work')
        tensor_names = build_simple_network(x, BATCH_SIZE, (256, 54))
    elif option == 'conv1':
        tensor_names = build_conv1(x, (128, 54))
    elif option == 'conv2':
        tensor_names = build_conv2(x, (128, 54), 32)
    else:
        print('INVALID OPTION')
        exit(1)

    logdir = 'experiments/VAE_baseline/{}'.format(option)

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

    options = {'simple', 'conv1', 'conv2'}

    args = argv[1:]
    assert len(args) == 1, 'You must provide one argument'
    option = args[0]
    assert option in options, 'options can be {}'.format(options)

    with tf.Graph().as_default():
        run_experiment(option)
