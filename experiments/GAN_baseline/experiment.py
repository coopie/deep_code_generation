import project_context  #  NOQA

from sys import argv
import numpy as np
import logging

from pipelines.one_hot_token import one_hot_token_pipeline
from pipelines.mnist import mnist_unlabeled_generator
from model_utils.queues import build_single_output_queue
from models import build_mnist_gan_for_training

import tensorflow as tf
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training.supervisor import Supervisor
from tensorflow.python.summary import summary
tf.logging.set_verbosity(tf.logging.INFO)


def run_experiment(option):
    BATCH_SIZE=32

    if option == 'mnist_digits':
        gen = mnist_unlabeled_generator(BATCH_SIZE, for_cnn=True)
        batch_shape = gen()[0].shape

        print('batch shape is : {}'.format(batch_shape))
        get_batch = lambda: gen()[0]

        queue = build_single_output_queue(get_batch, batch_shape)
        x = queue.dequeue(name='real_input')
        training_ops = build_mnist_gan_for_training(x)
    else:
        print('INVALID OPTION')
        exit(1)

    logdir = 'experiments/GAN_baseline/{}'.format(option)

    sv = Supervisor(
        logdir=logdir,
        save_summaries_secs=20,
        save_model_secs=120
    )
    # Get a TensorFlow session managed by the supervisor.
    with sv.managed_session() as sess:
        # Use the session to train the graph.

        d_loss = 5
        g_loss = 4
        steps_without_d_training = 0
        i = 0
        while not sv.should_stop():
            # if d_loss > 0.5 or steps_without_d_training > 50:
            if d_loss > g_loss or i > 6000:
                steps_without_d_training = 0
                d_loss = sess.run(training_ops['train_discriminator'])
            # else:
            #     steps_without_d_training += 1

            # sess.run(training_ops['train_discriminator'])
            g_loss = sess.run(training_ops['train_generator'])


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO
    )

    options = {'mnist_digits'}

    args = argv[1:]
    assert len(args) == 1, 'You must provide one argument'
    option = args[0]
    assert option in options, 'options can be {}'.format(options)

    with tf.Graph().as_default():
        run_experiment(option)
