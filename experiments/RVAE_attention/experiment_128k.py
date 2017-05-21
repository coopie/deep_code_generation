"""
Attention experiments:

Usage:
    experiment_128k.py [--basic] <option>
    experiment_128k.py -h | --help

Options:
    -h --help   Show this screen.
    -b --basic  Use the basic huzzer dataset.

"""

from sys import argv
import numpy as np
import os
from docopt import docopt

import logging
import project_context  # NOQA
from pipelines.one_hot_token import one_hot_token_random_batcher
from pipelines.data_sources import BASIC_DATASET_ARGS
from model_utils.queues import build_single_output_queue
from model_utils.loss_functions import kl_divergence, ce_loss_for_sequence_batch
from model_utils.ops import get_sequence_lengths, resampling
from models import (
    build_attention1_decoder,
    build_single_program_encoder,
)

import tensorflow as tf
from tensorflow.python.training.supervisor import Supervisor
tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim

BASEDIR = os.path.dirname(os.path.realpath(__file__))

TOKEN_EMB_SIZE = 54  # Using categorical labels for the finite subsetset of haskell
BATCH_SIZE = 128
NUMBER_BATCHES = 1000


def run_experiment(option, use_basic_dataset):
    sequence_cap = 56 if use_basic_dataset else 130
    print('Setting up data pipeline...')

    huzzer_kwargs = BASIC_DATASET_ARGS if use_basic_dataset else {}
    datasource = one_hot_token_random_batcher(
        BATCH_SIZE,
        NUMBER_BATCHES,
        length=sequence_cap,
        cache_path='attention_models_{}_{}_{}'.format(
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
    raw_input_sequences = queue.dequeue(name='input_sequence')
    sequence_lengths = get_sequence_lengths(
        tf.cast(raw_input_sequences, tf.int32)
    )
    input_sequences = tf.cast(raw_input_sequences, tf.float32)

    print('Building model..')
    if option.startswith('attention1'):
        z_size = int(option.split('_')[-1])
        encoder_output = build_single_program_encoder(input_sequences, sequence_lengths, z_size)
        z_resampled = resampling(encoder_output)
        decoder_output = build_attention1_decoder(
            z_resampled, sequence_lengths, sequence_cap, TOKEN_EMB_SIZE
        )
        cross_entropy_loss = tf.reduce_mean(
            ce_loss_for_sequence_batch(
                decoder_output,
                input_sequences,
                sequence_lengths,
                sequence_cap
            )
        )
        kl_loss = tf.reduce_mean(kl_divergence(encoder_output))
    else:
        print('INVALID OPTION')
        exit(1)

    total_loss_op = kl_loss + cross_entropy_loss
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    tf.summary.scalar('kl_loss', kl_loss)
    tf.summary.scalar('total_loss', total_loss_op)
    logdir = os.path.join(BASEDIR, ('basic_' if use_basic_dataset else '') + option)

    optimizer = tf.train.AdamOptimizer(1e-3)
    print('creating train op...')
    train_op = slim.learning.create_train_op(total_loss_op, optimizer)
    print('starting supervisor...')
    sv = Supervisor(
        logdir=logdir,
        save_model_secs=300,
        save_summaries_secs=60
    )
    print('training...')
    with sv.managed_session() as sess:
        while not sv.should_stop():
            total_loss, _ = sess.run([total_loss_op, train_op])


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
