import project_context  #  NOQA

from sys import argv
import numpy as np

import logging
from pipelines.one_hot_token import one_hot_variable_length_token_dataset
import tensorflow_fold as td

from models import (
    build_token_level_RVAE,
    build_train_graph_for_RVAE
)

import tensorflow as tf
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training.supervisor import Supervisor
from tensorflow.python.summary import summary
tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim


TOKEN_EMB_SIZE = 54  # Using categorical labels for the finite subsetset of haskell
NUM_STEPS_TO_STOP_IF_NO_IMPROVEMENT = 40
def run_experiment(option):
    BATCH_SIZE = 128
    NUMBER_BATCHES = 1000
    # set up pipeline
    print('Setting up data pipeline')
    dataset = one_hot_variable_length_token_dataset(
        batch_size=1,
        number_of_batches=BATCH_SIZE * 1000,
        cache_path='one_hot_token_variable_length_haskell_batch{}_number{}'.format(
            BATCH_SIZE, NUMBER_BATCHES
        )
    )

    # Generator that gets examples
    def get_example():
        while True:
            yield np.squeeze(dataset()[0], axis=0)

    # use the queue for training
    if option == 'gru_8':
        network_block = build_token_level_RVAE(32, TOKEN_EMB_SIZE)
        train_block = build_train_graph_for_RVAE(network_block)
    else:
        print('INVALID OPTION')
        exit(1)

    logdir = 'experiments/Recurrent_VAE_baseline/{}'.format(option)

    # compile and build the train op
    compiler = td.Compiler.create(train_block)

    metrics = compiler.metric_tensors
    kl_loss = tf.reduce_mean(metrics['kl_loss'])
    cross_entropy_loss = tf.reduce_mean(metrics['cross_entropy_loss'])
    total_loss = kl_loss + cross_entropy_loss
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    tf.summary.scalar('kl_loss', kl_loss)
    tf.summary.scalar('total_loss', total_loss)

    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = slim.learning.create_train_op(total_loss, optimizer)
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

            summary, global_step, total_loss _ = sess.run(
                [summary_op, sv.global_step, total_loss, train_op],
                feed_dict={compiler.loom_input_tensor: batch}
            )
            if i % steps_per_summary == 0:
                sv.summary_computed(sess, summary, global_step)

            # Stop if loss does not improve after 100 steps
            # add a small amount to the current loss to reduce noisey things
            if (total_loss + 0.005) < best_loss_so_far:
                best_loss_so_far = total_loss
                num_steps_until_best = 0
            else
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
