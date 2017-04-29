from sys import argv
import numpy as np
import os
from os.path import join
import tensorflow as tf
import tensorflow_fold as td

import project_context  # NOQA
from pipelines.one_hot_token import one_hot_variable_length_token_dataset
from models import (
    default_gru_cell,
    build_program_encoder,
    resampling
)

BASEDIR = os.path.dirname(os.path.realpath(__file__))


def analyze_model(option):
    TOKEN_EMB_SIZE = 54

    if option.startswith('single_layer_gru_blind_'):
        # TODO: this is old naming convention
        look_behind = 0
        z_size = int(option.split('_')[-1])
        directory = 'single_layer_gru_{}'.format(z_size)
        encoder_block = build_encoder(z_size, TOKEN_EMB_SIZE)
        # decoder_block = build_blind_decoder(z_size, TOKEN_EMB_SIZE)
        decoder_block = build_blind_decoder_block(
            z_size, TOKEN_EMB_SIZE
        )
        print('z_size={}'.format(z_size))
    else:
        exit('invalid option')

    print('Setting up data pipeline...')
    dataset = one_hot_variable_length_token_dataset(
        batch_size=1,
        number_of_batches=1000,
        cache_path='one_hot_token_variable_length_haskell_batch{}_number{}_lookbehind{}'.format(
            1, 1000, look_behind
        ),
        zero_front_pad=look_behind
    )

    def get_input():
        return np.squeeze(dataset()[0], axis=0)

    BASEDIR = 'experiments/Recurrent_VAE_baseline'
    path = join(BASEDIR, directory)

    encoder_graph = td.Compiler.create(encoder_block)
    (mus_and_log_sigs,) = encoder_graph.output_tensors

    decoder_graph = td.Compiler.create(decoder_block)
    un_normalised_token_probs, hidden_state = decoder_graph.output_tensors

    # build resampling op
    resampling_input, resample_op = build_resampling_op(z_size)

    saver = tf.train.Saver()

    examples = [get_input() for i in range(10)]
    ex = encoder_graph.build_loom_inputs(examples)

    with tf.Session() as sess:
        saver.restore(
            sess, tf.train.latest_checkpoint(path, 'checkpoint.txt')
        )
        for batch in td.group_by_batches(ex, 1):

            p_of_z = sess.run(mus_and_log_sigs, feed_dict={
                encoder_graph.loom_input_tensor: batch
            })
            resampled = sess.run(resample_op, feed_dict={
                resampling_input: p_of_z
            })

            # fd = encoder_graph.build_feed_dict([
            #     (resampled.squeeze(axis=0), [np.zeros((0,))])
            # ])
            fd = decoder_graph.build_feed_dict([
                (resampled.squeeze(axis=0), [np.zeros((0,))])
            ])

            x = sess.run(
                [un_normalised_token_probs, hidden_state],
                feed_dict=fd

            )

            import code  # NOQA
            code.interact(local=locals())


def build_encoder(z_size, token_emb_size):
    input_sequence = td.Map(td.Vector(token_emb_size))
    encoder_rnn_cell = build_program_encoder(default_gru_cell(2*z_size))
    output_sequence = td.RNN(encoder_rnn_cell) >> td.GetItem(0)
    mus_and_log_sigs = output_sequence >> td.GetItem(-1)
    return input_sequence >> mus_and_log_sigs


def build_blind_decoder_block(z_size, token_emb_size):

    c = td.Composition()
    c.set_input_type(
        td.TupleType(
            td.TensorType((z_size,)),
            td.SequenceType(td.TensorType((0,)))
        )
    )
    with c.scope():
        hidden_state = td.GetItem(0).reads(c.input)
        list_of_nothing = td.GetItem(1).reads(c.input)

        decoder_output = build_program_decoder_for_analysis(
            token_emb_size, default_gru_cell(z_size)
        )

        decoder_output.reads(list_of_nothing, hidden_state)
        decoder_rnn_output = td.GetItem(1).reads(decoder_output)
        un_normalised_token_probs = td.GetItem(0).reads(decoder_output)
        # get the fisrt output (meant to only compute one interation)
        c.output.reads(
            td.GetItem(0).reads(un_normalised_token_probs),
            td.GetItem(0).reads(decoder_rnn_output)
        )

    return td.Record((td.Vector(z_size), td.Map(td.Vector(0)))) >> c


def build_program_decoder_for_analysis(token_emb_size, rnn_cell):
    """
    Does the same as build_program_decoder_for_analysis, but also returns
        the final hidden state of the decoder
    """
    # c = td.Composition()
    # with c.scope():
    #     decoder_rnn = td.ScopedLayer(
    #         rnn_cell,
    #         'decoder'
    #     )
    #     decoder_rnn_output = td.RNN(
    #         decoder_rnn,
    #         initial_state_from_input=True
    #     ) >> td.GetItem(0)
    #
    #     fc_layer = td.FC(
    #         token_emb_size,
    #         activation=tf.nn.relu,
    #         initializer=tf.contrib.layers.xavier_initializer(),
    #         name='encoder_fc'
    #     )
    #     decoder_rnn_output.reads()
    #     un_normalised_token_probs = td.Map(fc_layer).reads(decoder_rnn_output)
    #     c.output.reads(un_normalised_token_probs, decoder_rnn_output)
    # return c
    decoder_rnn = td.ScopedLayer(
        rnn_cell,
        'decoder'
    )
    decoder_rnn_output = td.RNN(
        decoder_rnn,
        initial_state_from_input=True
    ) >> td.GetItem(0)

    fc_layer = td.FC(
        token_emb_size,
        activation=tf.nn.relu,
        initializer=tf.contrib.layers.xavier_initializer(),
        name='encoder_fc'
    )
    # decoder_rnn_output.reads()
    un_normalised_token_probs = td.Map(fc_layer)
    return decoder_rnn_output >> td.AllOf(un_normalised_token_probs, td.Identity())


def build_resampling_op(z_size):
    mus_and_log_sigs = tf.placeholder(tf.float32, (1, 2*z_size,))
    return mus_and_log_sigs, resampling(mus_and_log_sigs)



if __name__ == '__main__':
    args = argv[1:]
    assert len(args) == 1, 'must have exactly one argument'
    analyze_model(args[0])
