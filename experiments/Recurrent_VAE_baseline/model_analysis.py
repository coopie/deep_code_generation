from huzzer.tokenizing import TOKEN_MAP
from tqdm import tqdm
from sys import argv
import errno
import numpy as np
import os
from os.path import join
import tensorflow as tf
import tensorflow_fold as td
from scipy.misc import imsave

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
    NUMBER_OF_EXAMPLES = 100

    if option.startswith('single_layer_gru_blind_'):
        # TODO: this is old naming convention
        look_behind = 0
        z_size = int(option.split('_')[-1])
        directory = 'single_layer_gru_{}'.format(z_size)
        encoder_block = build_encoder(z_size, TOKEN_EMB_SIZE)
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
        return np.squeeze(dataset()[0], axis=0).astype(np.float32)

    path = join(BASEDIR, directory)

    encoder_graph = td.Compiler.create(encoder_block)
    (mus_and_log_sigs,) = encoder_graph.output_tensors

    decoder_graph = td.Compiler.create(decoder_block)
    un_normalised_token_probs, hidden_state_t = decoder_graph.output_tensors
    token_probs_t = tf.nn.softmax(un_normalised_token_probs)

    # Build resampling op
    resampling_input, resample_op = build_resampling_op(z_size)

    saver = tf.train.Saver()

    examples = [get_input() for i in range(NUMBER_OF_EXAMPLES)]
    fold_inputs = encoder_graph.build_loom_inputs(examples)

    sess = tf.Session()
    saver.restore(
        sess, tf.train.latest_checkpoint(path, 'checkpoint.txt')
    )
    # Autoencode_bit
    examples_dir = join(BASEDIR, option + '_examples')
    mkdir_p(examples_dir)

    autoencoded_examples_path = join(examples_dir, 'autoencoded')
    message = 'Autencoding {} examples to {}'.format(
        len(examples), autoencoded_examples_path
    )
    for i, (fold_batch, input_sequence) in enumerate(tqdm(
        zip(
            td.group_by_batches(fold_inputs, 1), examples
        ),
        desc=message, total=len(examples)
    )):
        dir_for_example = join(autoencoded_examples_path, str(i))
        mkdir_p(dir_for_example)

        p_of_z = sess.run(mus_and_log_sigs, feed_dict={
            encoder_graph.loom_input_tensor: fold_batch
        })
        mus = p_of_z[:, :z_size]

        fd = decoder_graph.build_feed_dict([
            (mus.squeeze(axis=0), [np.zeros((0,))])
        ])

        token_probs, hidden_state = sess.run(
            [token_probs_t, hidden_state_t],
            feed_dict=fd
        )
        input_code = example_to_code(input_sequence)
        write_to_file(join(dir_for_example, 'input.hs'), input_code)
        imsave(join(dir_for_example, 'input.png'), input_sequence.T)
        imsave(join(dir_for_example, 'z.png'), mus.reshape((mus.size // 32, 32)))

        # TODO: make this a random sample
        tokens_probs = [token_probs]
        text_so_far = [one_hot(TOKEN_EMB_SIZE, token_probs.squeeze().argmax())]
        for _ in range(len(input_sequence) - 1):
            fd = decoder_graph.build_feed_dict([
                (hidden_state.squeeze(), [np.zeros((0,))])
            ])

            token_probs, hidden_state = sess.run(
                [token_probs_t, hidden_state_t],
                feed_dict=fd
            )
            tokens_probs += [token_probs]
            text_so_far += [one_hot(TOKEN_EMB_SIZE, token_probs.squeeze().argmax())]

        decoder_output = np.concatenate(tokens_probs)
        imsave(join(dir_for_example, 'decoder_output.png'), decoder_output.T)
        write_to_file(join(dir_for_example, 'autoencoded_code.hs'), example_to_code(text_so_far))


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
        # get the first output (meant to only compute one interation)
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


def example_to_code(example):
    tokens = np.argmax(example, axis=-1)
    text = ' '.join([token_to_string(t) for t in tokens])
    return text


def token_to_string(t):
    if t == 0:
        return ''
    return TOKEN_MAP[t]


def write_to_file(path, text):
    with open(path, 'w') as f:
        f.write(text)


def one_hot(length, value):
    v = np.zeros(length, dtype=np.int8)
    v[value] = 1
    return v


# echoes the behaviour of mkdir -p
# from http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == '__main__':
    args = argv[1:]
    assert len(args) == 1, 'must have exactly one argument'
    analyze_model(args[0])
