"""
Attention experiments:

Usage:
    experiment_128k.py [--basic] <option>
    experiment_128k.py -h | --help

Options:
    -h --help   Show this screen.
    -b --basic  Use the basic huzzer dataset.

"""

import os
from docopt import docopt

import logging
import project_context  # NOQA
from pipelines.one_hot_token import one_hot_token_random_batcher
from pipelines.data_sources import BASIC_DATASET_ARGS
from model_utils.queues import build_single_output_queue
from model_utils.loss_functions import ce_loss_for_sequence_batch
from model_utils.ops import get_sequence_lengths
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
BATCH_SIZE = 32
NUMBER_BATCHES = 4000

# constants for BEGAN
LAMBDA = 0.001
GAMMA = 0.5


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
    real_sequence_lengths = get_sequence_lengths(
        tf.cast(raw_input_sequences, tf.int32)
    )
    real_input_sequences = tf.cast(raw_input_sequences, tf.float32)

    print('Building model..')
    if option.startswith('attention1'):
        z_size = int(option.split('_')[-1])

        random_vector = tf.random_normal(
            dtype=tf.float32,
            shape=[BATCH_SIZE, z_size],
            mean=0,
            stddev=0.1  # because that is what we will used when generating
        )

        # we do not know the length of the generated code beforehand, so we pass in
        # sequence lengths of `sequence_cap`
        full_lengths = tf.constant(
            [sequence_cap for _ in range(BATCH_SIZE)],
            dtype=tf.float32,
            name='generator_lengths'
        )

        # create the scaling const. k_t
        k_t = tf.Variable(0., trainable=False, name='k_t')

        # generator gets restored weights, and so does the
        with tf.variable_scope('generator'):
            unnormalized_generated_programs, _ = build_attention1_decoder(
                random_vector, full_lengths, sequence_cap, TOKEN_EMB_SIZE
            )
            generated_programs = tf.nn.softmax(
                unnormalized_generated_programs, dim=-1, name='generated_programs'
            )
            generated_lengths = get_sequence_lengths(generated_programs, epsilon=0.01)

        with tf.variable_scope('discriminator'):
            sequence_lengths = tf.concat([generated_lengths, real_sequence_lengths], axis=0)
            encoder_output = build_single_program_encoder(
                tf.concat([generated_programs, real_input_sequences], axis=0),
                sequence_lengths,
                z_size
            )
            # get the values corresponding to mus from the encoder output_shape
            assert encoder_output.get_shape()[1].value == 2 * z_size
            encoded_v = encoder_output[:, :z_size]
            reconstructed, _ = build_attention1_decoder(
                encoded_v, sequence_lengths, sequence_cap, TOKEN_EMB_SIZE
            )
            # these are the unnormalized_token_probs for g and d
            generated_reconstructed = reconstructed[:BATCH_SIZE]
            real_reconstructed = reconstructed[BATCH_SIZE:]

        generator_loss = tf.reduce_mean(
            ce_loss_for_sequence_batch(
                unnormalized_token_probs=generated_reconstructed,
                input_sequences=generated_programs,
                sequence_lengths=generated_lengths,
                max_length=sequence_cap
            )
        )
        real_loss = tf.reduce_mean(
            ce_loss_for_sequence_batch(
                unnormalized_token_probs=real_reconstructed,
                input_sequences=real_input_sequences,
                sequence_lengths=generated_lengths,
                max_length=sequence_cap
            )
        )
        discriminator_loss = real_loss - (k_t * generator_loss)

        optimizer = tf.train.AdamOptimizer(1e-5)
        print('creating discriminator train op...')
        d_train_op = slim.learning.create_train_op(discriminator_loss, optimizer)
        print('starting supervisor...')
        optimizer = tf.train.AdamOptimizer(1e-5)
        print('creating generator train op...')
        g_train_op = slim.learning.create_train_op(generator_loss, optimizer)

        balance = GAMMA * real_loss - generator_loss
        measure = real_loss + tf.abs(balance)

        # update k_t
        with tf.control_dependencies([d_train_op, g_train_op]):
            k_update = tf.assign(
                k_t, tf.clip_by_value(k_t + LAMBDA * balance, 0, 1))

        example_summary_op = tf.summary.merge([
            tf.summary.image("G", generated_programs),
            tf.summary.image("AE_G", tf.nn.softmax(generated_reconstructed, dim=-1)),
            tf.summary.image("AE_x", tf.nn.softmax(real_reconstructed, dim=-1)),
        ])

        perf_summary_op = tf.summary.merge([
            tf.summary.scalar("loss/d_loss", discriminator_loss),
            tf.summary.scalar("loss/d_loss_real", real_loss),
            tf.summary.scalar("loss/d_loss_fake", generator_loss),
            tf.summary.scalar("misc/measure", measure),
            tf.summary.scalar("misc/k_t", k_t),
            tf.summary.scalar("misc/balance", balance),
        ])

    else:
        print('INVALID OPTION')
        exit(1)

    logdir = os.path.join(BASEDIR, ('basic_' if use_basic_dataset else '') + option)

    sv = Supervisor(
        logdir=logdir,
        save_model_secs=300,
        save_summaries_secs=60,
        summary_op=perf_summary_op
    )
    print('training...')
    with sv.managed_session() as sess:
        i = 1
        while not sv.should_stop():
            i += 1
            ops = {
                'k_update': k_update,
                'measure': measure,
                'd_train_op': d_train_op,
                'g_train_op': g_train_op,
                'global_step': sv.global_step
            }
            if i % 200 == 0:
                ops.update({'images': example_summary_op})

            results = sess.run(ops)

            if i % 200 == 0:
                images_summary = results['images']
                global_step = results['global_step']
                sv.summary_writer.add_summary(images_summary, global_step)


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
