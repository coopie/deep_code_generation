import project_context  # NOQA
from sys import argv
import numpy as np
import os

import logging
from pipelines.one_hot_token import one_hot_token_random_batcher
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
NUM_STEPS_TO_STOP_IF_NO_IMPROVEMENT = 3000  # stop if no improvement after an epoch


def run_experiment(option):
    BATCH_SIZE = 128
    NUMBER_BATCHES = 1000
    SEQUENCE_CAP = 130
    print('Setting up data pipeline...')
    datasource = one_hot_token_random_batcher(
        BATCH_SIZE,
        NUMBER_BATCHES,
        length=SEQUENCE_CAP,
        cache_path='attention_models{}_{}'.format(NUMBER_BATCHES, BATCH_SIZE)
    )
    queue = build_single_output_queue(
        datasource,
        output_shape=(BATCH_SIZE, SEQUENCE_CAP, TOKEN_EMB_SIZE),
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
            z_resampled, sequence_lengths, SEQUENCE_CAP, TOKEN_EMB_SIZE
        )
        cross_entropy_loss = tf.reduce_mean(
            ce_loss_for_sequence_batch(
                decoder_output,
                input_sequences,
                sequence_lengths,
                SEQUENCE_CAP
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
    logdir = os.path.join(BASEDIR, option)

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
        best_loss_so_far = 100
        num_steps_until_best = 0

        i = -1
        while not sv.should_stop():
            i += 1

            total_loss, _ = sess.run([total_loss_op, train_op])
            # Stop if loss does not improve after some steps
            if total_loss < best_loss_so_far:
                best_loss_so_far = total_loss
                num_steps_until_best = 0
            else:
                num_steps_until_best += 1
                if num_steps_until_best == NUM_STEPS_TO_STOP_IF_NO_IMPROVEMENT:
                    exit()




if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO
    )
    args = argv[1:]
    assert len(args) == 1, 'You must provide one argument'
    option = args[0]

    run_experiment(option)
