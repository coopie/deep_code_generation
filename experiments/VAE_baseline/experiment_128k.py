"""
Baseline VAE experiments:
Experiments similar to those in experiment.py, but with limited dataset if 128000 examples
These are the final experiments done for the report

Usage:
    experiment_128k.py [--basic] <option>
    experiment_128k.py -h | --help

Options:
    -h --help   Show this screen.
    -b --basic  Use the basic huzzer dataset.


"""

from docopt import docopt
import logging

import project_context  # NOQA
from pipelines.data_sources import BASIC_DATASET_ARGS
from pipelines.one_hot_token import one_hot_token_random_batcher
from model_utils.queues import build_single_output_queue
from models import build_simple_network2

import tensorflow as tf
from tensorflow.python.training.supervisor import Supervisor
tf.logging.set_verbosity(tf.logging.INFO)


def run_experiment(option, use_basic_dataset):
    TOKEN_EMB_SIZE = 54
    BATCH_SIZE = 128
    if use_basic_dataset:
        sequence_cap = 130
    else:
        sequence_cap = 56

    X_SHAPE = (sequence_cap, TOKEN_EMB_SIZE)

    # set up pipeline
    print('Setting up data pipeline')
    NUMBER_BATCHES = 1000
    huzzer_kwargs = BASIC_DATASET_ARGS if use_basic_dataset else {}
    datasource = one_hot_token_random_batcher(
        BATCH_SIZE,
        NUMBER_BATCHES,
        length=sequence_cap,
        cache_path='simple_models_{}_{}_{}'.format(
            'basic' if use_basic_dataset else 'standard',
            NUMBER_BATCHES,
            BATCH_SIZE
        ),
        huzzer_kwargs=huzzer_kwargs
    )
    queue = build_single_output_queue(
        datasource,
        output_shape=(BATCH_SIZE, sequence_cap, TOKEN_EMB_SIZE),
        type=tf.uint8
    )

    raw_input_sequences = queue.dequeue(name='encoder_input')
    input_sequences = tf.cast(raw_input_sequences, tf.float32)
    if option.startswith('simple_'):
        z_size = int(option.split('_')[-1])
        build_simple_network2(
            input_sequences, X_SHAPE, latent_dim=z_size, kl_limit=0.0
        )
    else:
        print('INVALID OPTION')
        exit(1)

    logdir = 'experiments/VAE_baseline/{}{}'.format(
        'basic_' if use_basic_dataset else '',
        option
    )

    sv = Supervisor(
        logdir=logdir,
        save_summaries_secs=10,
        save_model_secs=120
    )
    # Get a TensorFlow session managed by the supervisor.
    with sv.managed_session() as sess:
        # Use the session to train the graph.
        for i in range(20000):
            if sv.should_stop():
                exit()
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

    args = docopt(__doc__, version='N/A')

    option = args.get('<option>')
    use_basic_dataset = args.get('--basic')
    run_experiment(option, use_basic_dataset)
