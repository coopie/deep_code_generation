import project_context  # NOQA
import tensorflow as tf
from tqdm import tqdm

from model_utils.loss_functions import kl_divergence
from model_utils.ops import vae_resampling



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
            attention_v = compute_attention_vector(
                hidden_states,
                unnormalized_attention_coefs
            )

        # Add h_i to hidden states
        hidden_states = tf.concat(
            (hidden_states, tf.expand_dims(h_state, 1)),
            axis=1
        )
        pbar.update(i+1)
    pbar.close()
    return tf.stack(unnormalized_token_probs, axis=1)


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


def calculate_ce_loss_for_batch(
    unnormalized_token_probs, input_sequences, sequence_lengths, max_length
):
    ce_losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=unnormalized_token_probs,
        labels=input_sequences
    )
    mask = tf.sequence_mask(sequence_lengths, max_length, dtype=tf.float32)

    masked = (mask * ce_losses)
    sums = tf.reduce_mean(masked, axis=-1)
    return sums


def compute_attention_vector(previous_hidden_states, unnormalized_attention_coefs):
    attention_coefs = tf.nn.softmax(
        unnormalized_attention_coefs, dim=-1
    )
    return tf.reduce_sum(
        tf.multiply(
            previous_hidden_states,
            tf.expand_dims(attention_coefs, -1)
        ),
        axis=1
    )


def simple_attention_coefs(previous_hidden_states, h, reuse, name_suffix=''):
    name_suffix = '_' + name_suffix if name_suffix else name_suffix
    h_proj = fully_connected(
        h,
        h.get_shape()[1].value,
        'simple_attention' + name_suffix,
        reuse=reuse
    )
    unnormalized_coefs = []
    for i in range(previous_hidden_states.get_shape()[1].value):
        h_prev = previous_hidden_states[:, i]
        unnormalized_coefs += [tf.reduce_sum(tf.multiply(h_proj, h_prev), axis=1)]

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
            (),
            initializer=initializer
        )
    return tf.matmul(input, weights) + bias


def get_batch_lengths(batch):
    is_padding = batch[:, :, 0]
    # add one to the value so the final 'nothing' character is added
    return (batch.shape[1] - is_padding.sum(axis=1)) + 1


def resampling(mus_and_log_sigs):
    """
    Batch resampling operation, mus_and_log_sigs of shape (b x z_size*2)
    """
    z_size = mus_and_log_sigs.get_shape()[-1].value // 2
    mus = mus_and_log_sigs[:, :z_size]
    log_sigs = mus_and_log_sigs[:, z_size:]

    z_resampled = vae_resampling(mus, log_sigs, epsilon_std=0.01)
    return z_resampled


def get_sequence_lengths(x):
    is_padding = x[:, :, 0]
    return (x.get_shape()[1].value - tf.reduce_sum(is_padding, axis=1)) + 1


def default_lstm_cell(size, activation=tf.tanh):
    return tf.contrib.rnn.LSTMCell(
        size,
        initializer=tf.contrib.layers.xavier_initializer(),
        activation=activation
    )
