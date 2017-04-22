import tensorflow as tf
import tensorflow_fold as td

# from pipelines.one_hot_token import one_hot_token_dataset

import numpy as np

from model_utils.ops import vae_resampling
from model_utils.loss_functions import vae_loss

# sess = tf.InteractiveSession()

# BATCH_SIZE = 128
# NUMBER_BATCHES = 500
# dataset = one_hot_token_dataset(
#     BATCH_SIZE,
#     NUMBER_BATCHES,
#     length=128,
#     cache_path='one_hot_token_haskell_batch{}_number{}'.format(BATCH_SIZE, NUMBER_BATCHES)
# )

# i = dataset()[0][0]

Z_SIZE = 32


def resampling(mus_and_log_sigs):
    z_size = mus_and_log_sigs.get_shape()[-1].value // 2
    mus = mus_and_log_sigs[:, :z_size]
    log_sigs = mus_and_log_sigs[:, z_size:]

    z_resampled = vae_resampling(mus, log_sigs, epsilon_std=0.01)
    return z_resampled


def null_tensor(*args):
    return tf.zeros((None,))


def build_VAE(z_size, token_emb_size):
    c = td.Composition()
    c.set_input_type(td.SequenceType(td.TensorType(([token_emb_size]), 'float32')))
    with c.scope():
        # input_sequence = td.Map(td.Vector(token_emb_size)).reads(c.input)
        input_sequence = c.input

        # encoder composition TODO: refactor this out
        # rnn_cell = td.ScopedLayer(
        #     tf.contrib.rnn.LSTMCell(
        #         num_units=2*z_size,
        #         initializer=tf.contrib.layers.xavier_initializer(),
        #         activation=tf.tanh
        #     ),
        #     'encoder'
        # )
        encoder_rnn_cell = td.ScopedLayer(
            tf.contrib.rnn.GRUCell(
                num_units=2*z_size,
                # initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.tanh
            ),
            'encoder'
        )
        output_sequence = td.RNN(encoder_rnn_cell) >> td.GetItem(0)
        mus_and_log_sigs = output_sequence >> td.GetItem(-1)

        # reparam_z = mus_and_log_sigs >> td.Function(resampling)
        reparam_z = td.Function(resampling, name='resampling')
        reparam_z.set_input_type(td.TensorType((2 * z_size,)))
        reparam_z.set_output_type(td.TensorType((z_size,)))

        #  A list of same length of input_sequence, but with empty values
        #  this is used for the decoder to map over
        list_of_nothing = td.Map(
            td.Void() >> td.FromTensor(tf.zeros((0,)))
        )

        # decoder composition
        # TODO: refactor this out
        # decoder_rnn = td.ScopedLayer(
        #     tf.contrib.rnn.LSTMCell(
        #         num_units=z_size,
        #         initializer=tf.contrib.layers.xavier_initializer(),
        #         activation=tf.tanh
        #     ),
        #     'decoder'
        # )
        decoder_rnn = td.ScopedLayer(
            tf.contrib.rnn.GRUCell(
                num_units=z_size,
                # initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.tanh
            ),
            'decoder'
        )
        decoder_rnn_output = td.RNN(
            decoder_rnn,
            initial_state_from_input=True
        ) >> td.GetItem(0)

        fc_layer = td.FC(
            token_emb_size,
            activation=tf.nn.relu,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        un_normalised_token_probs = decoder_rnn_output >> td.Map(fc_layer)

        # reparam_z.reads(input_sequence)
        mus_and_log_sigs.reads(input_sequence)
        reparam_z.reads(mus_and_log_sigs)
        list_of_nothing.reads(input_sequence)
        un_normalised_token_probs.reads(list_of_nothing, reparam_z)

        c.output.reads(un_normalised_token_probs, mus_and_log_sigs)
    return c


def softmax_crossentropy(logit, label_vector):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels=label_vector,
        logits=logit
    )


def kl_divergence(mus_and_log_sigs):

    halfway = int(mus_and_log_sigs.get_shape()[1].value / 2)  # HACK: make this cleaner
    mus = mus_and_log_sigs[:, :halfway]
    log_sigs = mus_and_log_sigs[:, halfway:]

    kl_loss_term = -0.5 * tf.reduce_mean(
        1 + log_sigs - tf.square(mus) - tf.exp(log_sigs),
        axis=1
    )

    return kl_loss_term


c = td.Composition()
with c.scope():
    input_sequence = td.Map(td.Vector(54)).reads(c.input)

    # net = build_VAE(Z_SIZE, 54)
    # un_normalised_token_probs, mus_and_log_sigs = input_sequence >> build_VAE(Z_SIZE, 54)
    network_output = build_VAE(Z_SIZE, 54)

    network_output.reads(input_sequence)

    un_normalised_token_probs = td.GetItem(0).reads(network_output)
    mus_and_log_sigs = td.GetItem(1).reads(network_output)

    cross_entropy_loss = td.ZipWith(td.Function(softmax_crossentropy)) >> td.Mean()
    cross_entropy_loss.reads(
        un_normalised_token_probs,
        input_sequence
    )
    kl_loss = td.Function(kl_divergence)
    kl_loss.reads(mus_and_log_sigs)

    td.Metric('cross_entropy_loss').reads(cross_entropy_loss)
    td.Metric('kl_loss').reads(kl_loss)

    c.output.reads(td.Void())



#  Tokenised version of my code
example_input = np.array([
    1,  2, 51, 16,  4, 17, 52,  3, 53, 16,  5, 38,  6, 37,  6, 37,  6,
    37,  6, 38,  6, 37,  6, 37,  6, 38, 53, 16,  8,  9, 10, 11, 12, 13,
    14,  7, 51, 17, 11, 48, 11,  8, 52, 53, 17,  5, 37,  6, 38,  6, 37,
    6, 38,  6, 38, 53, 17,  8,  9, 10, 11,  7, 51,  9, 26, 51, 20,  9,
    9, 52, 52,  0
])


def one_hotify(code_rep):
    one_hots = np.zeros((len(code_rep), 54), dtype='int8')
    one_hots[range(len(code_rep)), code_rep] = 1
    return one_hots


compiler = td.Compiler.create(c)
metrics = compiler.metric_tensors
kl_loss = metrics['kl_loss']
cross_entropy_loss = metrics['cross_entropy_loss']


loss = tf.reduce_mean(kl_loss + cross_entropy_loss)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

ex = one_hotify(example_input)
fd = compiler.build_feed_dict([ex], batch_size=1)

sess = tf.Session()
with sess.as_default():
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    x = list(fd.values())[0]
    import code  # NOQA
    code.interact(local=locals())

    kl, ce, _ = sess.run([kl_loss, cross_entropy_loss, train_op], feed_dict=fd)
