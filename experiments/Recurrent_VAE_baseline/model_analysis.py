""" Model Analysis for tf-fold RVAES.

Usage:
  model_analysis.py [-ag] <experiment>
  model_analysis.py -h | --help
  model_analysis.py --version

Options:
  -h --help       Show this screen.
  --version       Show version.
  -a              Skip autoencoding step.
  -g              Skip generating step.
"""

from docopt import docopt
from huzzer.tokenizing import TOKEN_MAP
from tqdm import tqdm, trange
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
    build_program_decoder,
    resampling
)

BASEDIR = os.path.dirname(os.path.realpath(__file__))
TOKEN_EMB_SIZE = 54
NUMBER_OF_EXAMPLES = 1000
MAX_PROGRAM_LENGTH = 250


def analyze_model(option, autoencode, generate):

    if option.startswith('single_layer_gru_blind_'):
        # TODO: this is old naming convention
        look_behind = 0
        z_size = int(option.split('_')[-1])
        directory = 'single_layer_gru_{}'.format(z_size)
        encoder_block = build_encoder(z_size, TOKEN_EMB_SIZE)

        decoder_rnn_block = build_program_decoder_for_analysis(
            TOKEN_EMB_SIZE, default_gru_cell(z_size)
        )

        decoder_block = build_decoder_block_for_analysis(
            z_size, TOKEN_EMB_SIZE, decoder_rnn_block, 0
        )
        print('z_size={}'.format(z_size))
    if option.startswith('single_layer_gru_look_behind_'):
        look_behind = int(option.split('_')[-2])
        z_size = int(option.split('_')[-1])
        directory = option
        encoder_block = build_encoder(z_size, TOKEN_EMB_SIZE)
        decoder_rnn_block = build_program_decoder(
            TOKEN_EMB_SIZE, default_gru_cell(z_size)
        )
        decoder_block = build_decoder_block_for_analysis(
            z_size,
            TOKEN_EMB_SIZE,
            decoder_rnn_block,
            TOKEN_EMB_SIZE * look_behind
        )
        print('z_size={}, look_behind={}'.format(z_size, look_behind))
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

    examples = [get_input() for i in range(NUMBER_OF_EXAMPLES)]
    fold_inputs = encoder_graph.build_loom_inputs(examples)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(
        sess, tf.train.latest_checkpoint(path, 'checkpoint.txt')
    )

    if autoencode:
        autoencode_procedure(
            option, path, z_size, examples, fold_inputs,
            encoder_graph, decoder_graph, mus_and_log_sigs, look_behind,
            token_probs_t, hidden_state_t, sess
        )
    if generate:
        generate_procedure(
            option, path, z_size, examples, fold_inputs,
            encoder_graph, decoder_graph, mus_and_log_sigs, look_behind,
            token_probs_t, hidden_state_t, sess
        )


def generate_procedure(
    option, path, z_size, examples, fold_inputs,
    encoder_graph, decoder_graph, mus_and_log_sigs, look_behind,
    token_probs_t, hidden_state_t, sess
):
    # Autoencode_bit
    examples_dir = join(BASEDIR, option + '_examples')
    mkdir_p(examples_dir)
    generated_examples_path = join(examples_dir, 'generated')
    message = 'Generating {} examples'.format(
        len(examples)
    )
    for i in trange(NUMBER_OF_EXAMPLES, desc=message):
        dir_for_example = join(generated_examples_path, str(i))
        mkdir_p(dir_for_example)

        # DECODER PART
        if look_behind > 0:
            output_sequence = []
            decoder_input = padded_look_behind_generator_generator(
                output_sequence, look_behind, TOKEN_EMB_SIZE
            )
        else:
            output_sequence = []
            decoder_input = zeros_generator()  # NOQA

        z_gen = np.random.normal(0, 1, z_size)
        imsave(join(dir_for_example, 'z.png'), z_gen.reshape((z_gen.size // 32, 32)))

        fd = decoder_graph.build_feed_dict([
            (z_gen, [next(decoder_input)])
        ])

        token_probs, hidden_state = sess.run(
            [token_probs_t, hidden_state_t],
            feed_dict=fd
        )

        # take a sample from the token probs
        sampled_token = np.random.multinomial(1, token_probs.squeeze() * 0.999)
        output_sequence.append(np.expand_dims(sampled_token, 0))

        for _ in range(MAX_PROGRAM_LENGTH - 1):
            fd = decoder_graph.build_feed_dict([
                (hidden_state.squeeze(), [next(decoder_input)])
            ])

            token_probs, hidden_state = sess.run(
                [token_probs_t, hidden_state_t],
                feed_dict=fd
            )
            sampled_token = np.random.multinomial(1, token_probs.squeeze() * 0.999)
            output_sequence.append(np.expand_dims(sampled_token, 0))
            if sampled_token[0]:
                break

        output_sequence = output_sequence[look_behind:]
        output_sequence = np.concatenate(output_sequence)
        imsave(join(dir_for_example, 'decoder_output.png'), output_sequence.T)

        write_to_file(
            join(dir_for_example, 'generated_code.hs'),
            example_to_code(output_sequence)
        )


def autoencode_procedure(
    option, path, z_size, examples, fold_inputs,
    encoder_graph, decoder_graph, mus_and_log_sigs, look_behind,
    token_probs_t,
    hidden_state_t,
    sess

):
    # Autoencode_bit
    examples_dir = join(BASEDIR, option + '_examples')
    mkdir_p(examples_dir)

    autoencoded_examples_path = join(examples_dir, 'autoencoded')
    message = 'Autencoding {} examples'.format(
        len(examples)
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

        input_code = example_to_code(input_sequence[look_behind:])
        write_to_file(join(dir_for_example, 'input.hs'), input_code)
        imsave(join(dir_for_example, 'input.png'), input_sequence.T)
        imsave(join(dir_for_example, 'z.png'), mus.reshape((mus.size // 32, 32)))

        # DECODER PART
        if look_behind > 0:
            decoder_input = padded_look_behind_generator(
                input_sequence, look_behind, TOKEN_EMB_SIZE
            )
        else:
            decoder_input = zeros_generator()  # NOQA

        fd = decoder_graph.build_feed_dict([
            (mus.squeeze(axis=0), [next(decoder_input)])
        ])

        token_probs, hidden_state = sess.run(
            [token_probs_t, hidden_state_t],
            feed_dict=fd
        )

        tokens_probs = [token_probs]
        text_so_far = [one_hot(TOKEN_EMB_SIZE, token_probs.squeeze().argmax())]
        for _ in range(len(input_sequence) - 1):
            fd = decoder_graph.build_feed_dict([
                (hidden_state.squeeze(), [next(decoder_input)])
            ])

            token_probs, hidden_state = sess.run(
                [token_probs_t, hidden_state_t],
                feed_dict=fd
            )
            tokens_probs += [token_probs]
            text_so_far += [one_hot(TOKEN_EMB_SIZE, token_probs.squeeze().argmax())]

        decoder_output = np.concatenate(tokens_probs)
        imsave(join(dir_for_example, 'decoder_output.png'), decoder_output.T)
        write_to_file(
            join(dir_for_example, 'autoencoded_code.hs'),
            example_to_code(text_so_far)
        )


def build_encoder(z_size, token_emb_size):
    input_sequence = td.Map(td.Vector(token_emb_size))
    encoder_rnn_cell = build_program_encoder(default_gru_cell(2*z_size))
    output_sequence = td.RNN(encoder_rnn_cell) >> td.GetItem(0)
    mus_and_log_sigs = output_sequence >> td.GetItem(-1)
    return input_sequence >> mus_and_log_sigs


def build_decoder_block_for_analysis(z_size, token_emb_size, decoder_cell, input_size):

    c = td.Composition()
    c.set_input_type(
        td.TupleType(
            td.TensorType((z_size,)),
            td.SequenceType(td.TensorType((input_size,)))
        )
    )
    with c.scope():
        hidden_state = td.GetItem(0).reads(c.input)
        rnn_input = td.GetItem(1).reads(c.input)

        # decoder_output = build_program_decoder_for_analysis(
        #     token_emb_size, default_gru_cell(z_size)
        # )
        decoder_output = decoder_cell

        decoder_output.reads(rnn_input, hidden_state)
        decoder_rnn_output = td.GetItem(1).reads(decoder_output)
        un_normalised_token_probs = td.GetItem(0).reads(decoder_output)
        # get the first output (meant to only compute one interation)
        c.output.reads(
            td.GetItem(0).reads(un_normalised_token_probs),
            td.GetItem(0).reads(decoder_rnn_output)
        )

    return td.Record(
        (td.Vector(z_size), td.Map(td.Vector(input_size)))
    ) >> c


# this needs changing to somehting like "build blind decoder"
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


def zeros_generator():
    while True:
        yield np.zeros((0,))


def padded_look_behind_generator(
    input_sequence, look_behind, token_emb_size
):
    num_inputs = len(input_sequence)
    zero_pad = np.zeros((look_behind, token_emb_size))
    input_sequence = np.concatenate((zero_pad, input_sequence))
    for i in range(num_inputs):
        yield input_sequence[i: i + look_behind].flatten()


def padded_look_behind_generator_generator(
    token_list, look_behind, token_emb_size
):
    token_list.extend([np.zeros((1, TOKEN_EMB_SIZE)) for x in range(look_behind)])
    i = -1
    while True:
        i += 1
        yield np.concatenate(token_list[i: i + look_behind]).flatten()


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
    args = docopt(__doc__, version='0.0.1')
    analyze_model(
        args.get('<experiment>'),
        not args.get('-a'),
        not args.get('-g')
    )
