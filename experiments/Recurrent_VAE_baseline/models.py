import project_context  # NOQA
import tensorflow as tf
import tensorflow_fold as td

from model_utils.loss_functions import kl_divergence, softmax_crossentropy
from model_utils.ops import vae_resampling


def resampling(mus_and_log_sigs):
    """
    Batch resampling operation, mus_and_log_sigs of shape (b x z_size*2)
    """
    z_size = mus_and_log_sigs.get_shape()[-1].value // 2
    mus = mus_and_log_sigs[:, :z_size]
    log_sigs = mus_and_log_sigs[:, z_size:]

    z_resampled = vae_resampling(mus, log_sigs, epsilon_std=0.01)
    return z_resampled


def resampling_block(z_size):
    reparam_z = td.Function(resampling, name='resampling')
    reparam_z.set_input_type(td.TensorType((2 * z_size,)))
    reparam_z.set_output_type(td.TensorType((z_size,)))
    return reparam_z


def default_gru_cell(size, activation=tf.tanh):
    return tf.contrib.rnn.GRUCell(
        num_units=size,
        activation=activation
    )


def default_lstm_cell(size):
    return tf.contrib.rnn.BasicLSTMCell(
        num_units=size,
        activation=tf.tanh
    )


def build_program_encoder(rnn_cell):
    return td.ScopedLayer(
        rnn_cell,
        'encoder'
    )


def build_program_decoder(token_emb_size, rnn_cell, just_tokens=False):
    """
    Used for blind or 'look-behind' decoders
    """
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
        name='encoder_fc'  # this is fantastic
    )

    # un_normalised_token_probs = decoder_rnn_output >> td.Map(fc_layer)
    if just_tokens:
        return decoder_rnn_output >> td.Map(fc_layer)
    else:
        return decoder_rnn_output >> td.AllOf(td.Map(fc_layer), td.Identity())
    # return un_normalised_token_probs


def build_token_level_RVAE(z_size, token_emb_size, look_behind_length):
    c = td.Composition()
    c.set_input_type(td.SequenceType(td.TensorType(([token_emb_size]), 'float32')))
    with c.scope():
        padded_input_sequence = c.input
        # build encoder block
        encoder_rnn_cell = build_program_encoder(default_gru_cell(2 * z_size))

        output_sequence = td.RNN(encoder_rnn_cell) >> td.GetItem(0)
        mus_and_log_sigs = output_sequence >> td.GetItem(-1)

        reparam_z = resampling_block(z_size)

        if look_behind_length > 0:
            decoder_input_sequence = (
                td.Slice(stop=-1) >> td.NGrams(look_behind_length) >> td.Map(td.Concat())
            )
        else:
            decoder_input_sequence = td.Map(
                td.Void() >> td.FromTensor(tf.zeros((0,)))
            )

        # build decoder block
        un_normalised_token_probs = build_program_decoder(
            token_emb_size, default_gru_cell(z_size), just_tokens=True
        )

        # remove padding for input sequence
        input_sequence = td.Slice(start=look_behind_length)
        input_sequence.reads(padded_input_sequence)

        mus_and_log_sigs.reads(input_sequence)
        reparam_z.reads(mus_and_log_sigs)

        decoder_input_sequence.reads(padded_input_sequence)
        td.Metric('encoder_sequence_length').reads(td.Length().reads(input_sequence))
        td.Metric('decoder_sequence_length').reads(td.Length().reads(decoder_input_sequence))
        un_normalised_token_probs.reads(decoder_input_sequence, reparam_z)

        c.output.reads(un_normalised_token_probs, mus_and_log_sigs)
    return c


# def build_token_level_RVAE_no_look_behind(z_size, token_emb_size):
#     # Curently uses only GRU
#     c = td.Composition()
#     c.set_input_type(td.SequenceType(td.TensorType(([token_emb_size]), 'float32')))
#     with c.scope():
#         input_sequence = c.input
#         # build encoder block
#         encoder_rnn_cell = build_program_encoder(default_gru_cell(2*z_size))
#
#         output_sequence = td.RNN(encoder_rnn_cell) >> td.GetItem(0)
#         mus_and_log_sigs = output_sequence >> td.GetItem(-1)
#
#         reparam_z = resampling_block(z_size)
#
#         #  A list of same length of input_sequence, but with empty values
#         #  this is used for the decoder to map over
#         list_of_nothing = td.Map(
#             td.Void() >> td.FromTensor(tf.zeros((0,)))
#         )
#
#         # build decoder block
#         un_normalised_token_probs = build_program_decoder(
#             token_emb_size, default_gru_cell(z_size)
#         )
#
#         mus_and_log_sigs.reads(input_sequence)
#         reparam_z.reads(mus_and_log_sigs)
#         list_of_nothing.reads(input_sequence)
#         un_normalised_token_probs.reads(list_of_nothing, reparam_z)
#
#         c.output.reads(un_normalised_token_probs, mus_and_log_sigs)
#     return c


# def build_token_level_RVAE_look_behind(z_size, token_emb_size, look_behind_length):
    # currently only uses GRU cells
    # c = td.Composition()
    # c.set_input_type(td.SequenceType(td.TensorType(([token_emb_size]), 'float32')))
    # with c.scope():
    #     padded_input_sequence = c.input
    #     # build encoder block
    #     encoder_rnn_cell = build_program_encoder(default_gru_cell(2 * z_size))
    #
    #     output_sequence = td.RNN(encoder_rnn_cell) >> td.GetItem(0)
    #     mus_and_log_sigs = output_sequence >> td.GetItem(-1)
    #
    #     reparam_z = resampling_block(z_size)
    #
    #     decoder_look_behind = (
    #         td.Slice(stop=-1) >> td.NGrams(look_behind_length) >> td.Map(td.Concat())
    #     )
    #     # build decoder block
    #     un_normalised_token_probs = build_program_decoder(
    #         token_emb_size, default_gru_cell(z_size)
    #     )
    #
    #     # remove padding for input sequence
    #     input_sequence = td.Slice(start=look_behind_length)
    #     input_sequence.reads(padded_input_sequence)
    #
    #     mus_and_log_sigs.reads(input_sequence)
    #     reparam_z.reads(mus_and_log_sigs)
    #
    #     decoder_look_behind.reads(padded_input_sequence)
    #     td.Metric('encoder_sequence_length').reads(td.Length().reads(input_sequence))
    #     td.Metric('decoder_sequence_length').reads(td.Length().reads(decoder_look_behind))
    #     un_normalised_token_probs.reads(decoder_look_behind, reparam_z)
    #
    #     c.output.reads(un_normalised_token_probs, mus_and_log_sigs)
    # return c


def build_train_graph_for_RVAE(rvae_block, look_behind_length=0):
    token_emb_size = get_size_of_input_vecotrs(rvae_block)

    c = td.Composition()
    with c.scope():
        padded_input_sequence = td.Map(td.Vector(token_emb_size)).reads(c.input)
        network_output = rvae_block
        network_output.reads(padded_input_sequence)

        un_normalised_token_probs = td.GetItem(0).reads(network_output)
        mus_and_log_sigs = td.GetItem(1).reads(network_output)

        input_sequence = td.Slice(start=look_behind_length).reads(padded_input_sequence)
        # TODO: metric that output of rnn is the same as input sequence
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

    return c


def get_size_of_input_vecotrs(block):
    input_type = block.input_type
    assert type(input_type) == td.SequenceType
    input_tensor_shape = input_type.element_type.shape
    assert len(input_tensor_shape) == 1, 'input_to_network needs to be list of vectors'
    token_emb_size = input_tensor_shape[0]
    return token_emb_size
