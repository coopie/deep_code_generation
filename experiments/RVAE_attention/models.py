import project_context  # NOQA
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from model_utils.ops import (
    resampling,
    compute_attention_vector
)



def build_RVAE_model():
    pass


def build_attention1_decoder(z, sequence_lengths, max_length, token_emb_size):
    batch_size = tf.shape(z)[0]
    z_size = z.get_shape()[1].value

    rnn_cell = default_lstm_cell(z_size, tf.tanh)

    # raw output of the decoder
    unnormalized_token_probs = []
    # storage for the hidden states of the decoder
    hidden_states = tf.expand_dims(z, 1)

    h_state = z
    c_state = rnn_cell.zero_state(batch_size, dtype=tf.float32).c
    attention_v = tf.zeros(tf.shape(z))

    attention_weights = []

    pbar = tqdm(desc='Building decoder ops', total=sum(range(max_length+1)))
    for i in range(max_length):
        # compute h_{i+1}
        with tf.variable_scope('decoder_rnn', reuse=i > 0):
            unused, (c_state, h_state) = rnn_cell(attention_v, (c_state, h_state))

        # compute t_{i+1}
        unnormalized_token_prob = fully_connected(
            h_state,
            token_emb_size,
            'decoder_fully_connected',
            reuse=i > 0
        )
        unnormalized_token_probs += [unnormalized_token_prob]

        # Compute a_{i+1} from f([h_0 ... h_{i}], h_{i+1})
        if i < (max_length - 1):
            unnormalized_attention_coefs = simple_attention_coefs(
                hidden_states,
                h_state,
                reuse=i > 0
            )
            attention_v, weights = compute_attention_vector(
                hidden_states,
                unnormalized_attention_coefs
            )
            attention_weights += [weights]

        # Add h_i to hidden states
        hidden_states = tf.concat(
            (hidden_states, tf.expand_dims(h_state, 1)),
            axis=1
        )
        pbar.update(i+1)
    pbar.close()
    return tf.stack(unnormalized_token_probs, axis=1), attention_weights


def build_single_program_encoder(input_sequences, sequence_lengths, z_size):
    """
    May be used for bi directional (if used also on the reverse of the input sequences)
    """
    rnn_cell = default_lstm_cell(2*z_size, activation=tf.tanh)
    outputs, (c_state, m_state) = tf.nn.dynamic_rnn(
        rnn_cell,
        dtype=tf.float32,
        sequence_length=sequence_lengths,
        inputs=input_sequences
    )
    return m_state


def simple_attention_coefs(previous_hidden_states, h, reuse, name_suffix=''):
    """
    computes the cosine similarity beween h^T * W and all previous hidden_states
    """
    name_suffix = '_' + name_suffix if name_suffix else name_suffix
    h_proj = fully_connected(
        h,
        h.get_shape()[1].value,
        'simple_attention' + name_suffix,
        reuse=reuse
    )
    h_proj_normalized = tf.nn.l2_normalize(h_proj, -1)

    unnormalized_coefs = []
    for i in range(previous_hidden_states.get_shape()[1].value):
        h_prev = previous_hidden_states[:, i]
        h_prev_normalized = tf.nn.l2_normalize(h_prev, -1)
        unscaled_coef = tf.reduce_sum(
            tf.multiply(h_proj_normalized, h_prev_normalized), axis=1
        )

        # Use the dynamic sofmax with m=sqrt(l) and epsilon=0.001
        l = previous_hidden_states.get_shape()[1].value
        if l > 1:
            epsilon = 0.001
            m = np.sqrt(l)
            dynamic_scaling_factor = 0.5 * np.log(
                ((1 - epsilon) * (l - m)) / (epsilon * m)
            )
        else:
            dynamic_scaling_factor = 1

        unnormalized_coefs += [dynamic_scaling_factor * unscaled_coef]
    return tf.stack(
        unnormalized_coefs, axis=1
    )


def fully_connected(
    input, output_size,
    var_name_scope,
    reuse=None,
    initializer=tf.contrib.layers.xavier_initializer()
):
    assert reuse is not None, 'Must set reuse value'

    input_size = input.get_shape()[-1].value
    with tf.variable_scope(var_name_scope, reuse=reuse):
        weights = tf.get_variable(
            'weights',
            (input_size, output_size),
            initializer=initializer
        )
        bias = tf.get_variable(
            'bias',
            (output_size,),
            initializer=initializer
        )
    return tf.matmul(input, weights) + bias


def default_lstm_cell(size, activation=tf.tanh):
    return tf.contrib.rnn.LSTMCell(
        size,
        initializer=tf.contrib.layers.xavier_initializer(),
        activation=activation
    )
