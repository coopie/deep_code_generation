"""
Baseline RVAE experiments:
Experiments using a limited dataset if 128000 examples

Usage:
    experiment_128k.py [--basic] <option>
    experiment_128k.py -h | --help

Options:
    -h --help   Show this screen.
    -b --basic  Use the basic huzzer dataset.


"""
from docopt import docopt
from sys import argv
import numpy as np

import logging
import project_context  # NOQA
from pipelines.one_hot_token import one_hot_variable_length_token_dataset
from pipelines.data_sources import BASIC_DATASET_ARGS
import tensorflow_fold as td

from models import (
    build_token_level_RVAE,
    build_train_graph_for_RVAE
)

import tensorflow as tf
from tensorflow.python.training.supervisor import Supervisor
tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim


TOKEN_EMB_SIZE = 54  # Using categorical labels for the finite subsetset of haskell
NUM_STEPS_TO_STOP_IF_NO_IMPROVEMENT = 3000  # stop if no improvement after an epoch


def run_experiment(option, use_basic_dataset):
    BATCH_SIZE = 128
    NUMBER_BATCHES = 1000

    print('Building model..')
    if option.startswith('single_layer_gru_blind_'):
        look_behind = 0
        num_grus = int(option.split('_')[-1])
        network_block = build_token_level_RVAE(num_grus, TOKEN_EMB_SIZE, look_behind_length=0)
        train_block = build_train_graph_for_RVAE(network_block)
    elif option.startswith('single_layer_gru_look_behind_'):
        num_grus = int(option.split('_')[-1])
        look_behind = int(option.split('_')[-2])
        network_block = build_token_level_RVAE(
            num_grus, TOKEN_EMB_SIZE, look_behind
        )
        train_block = build_train_graph_for_RVAE(network_block, look_behind)
    else:
        print('INVALID OPTION')
        exit(1)

    print('Setting up data pipeline...')
    huzzer_kwargs = BASIC_DATASET_ARGS if use_basic_dataset else {}
    # the generator for fold needs one example at a time,
    dataset = one_hot_variable_length_token_dataset(
        batch_size=1,
        number_of_batches=BATCH_SIZE * NUMBER_BATCHES,
        cache_path='one_hot_token_variable_length_haskell_batch{}_number{}_lookbehind{}'.format(
            1, NUMBER_BATCHES * BATCH_SIZE, look_behind
        ),
        zero_front_pad=look_behind,
        huzzer_kwargs=huzzer_kwargs
    )

    # Generator that gets examples
    def get_example():
        while True:
            yield np.squeeze(dataset()[0], axis=0)

    logdir = 'experiments/Recurrent_VAE_baseline/{}{}'.format(
        'basic_' if use_basic_dataset else '', option
    )

    # compile and build the train op
    compiler = td.Compiler.create(train_block)

    metrics = compiler.metric_tensors
    kl_loss = tf.reduce_mean(metrics['kl_loss'])
    cross_entropy_loss = tf.reduce_mean(metrics['cross_entropy_loss'])
    total_loss_op = kl_loss + cross_entropy_loss
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    tf.summary.scalar('kl_loss', kl_loss)
    tf.summary.scalar('total_loss', total_loss_op)

    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = slim.learning.create_train_op(total_loss_op, optimizer)
    summary_op = tf.summary.merge_all()

    sv = Supervisor(
        logdir=logdir,
        save_model_secs=60,
        summary_op=None,
    )
    print('training...')
    with sv.managed_session() as sess:

        batcher = compiler.build_loom_input_batched(get_example(), BATCH_SIZE)

        steps_per_summary = 10
        best_loss_so_far = 100
        num_steps_until_best = 0

        for i, batch in enumerate(batcher):
            if sv.should_stop():
                break

            encoder_sequence_length_t = compiler.metric_tensors['encoder_sequence_length']
            decoder_sequence_length_t = compiler.metric_tensors['decoder_sequence_length']

            le, ld, summary, global_step, total_loss, _ = sess.run(
                [
                    encoder_sequence_length_t,
                    decoder_sequence_length_t,
                    summary_op,
                    sv.global_step,
                    total_loss_op,
                    train_op
                ],
                feed_dict={compiler.loom_input_tensor: batch}
            )
            assert all(le == ld), \
                'the encoder is folding over a different length sequence to encoder'
            if i % steps_per_summary == 0:
                sv.summary_computed(sess, summary, global_step)

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
    args = docopt(__doc__, version='N/A')

    option = args.get('<option>')
    use_basic_dataset = args.get('--basic')
    run_experiment(option, use_basic_dataset)
