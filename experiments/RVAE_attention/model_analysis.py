"""
Model analysis for attention experiments:

Usage:
    experiment_128k.py [--basic] <option>
    experiment_128k.py -h | --help

Options:
    -h --help   Show this screen.
    -b --basic  Use the basic huzzer dataset.

"""

from docopt import docopt
from huzzer.tokenizing import TOKEN_MAP
from tqdm import tqdm
import errno
import numpy as np
import os
from os.path import join
import tensorflow as tf
from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib

import project_context  # NOQA
from pipelines.data_sources import BASIC_DATASET_ARGS
from pipelines.one_hot_token import one_hot_token_dataset
from model_utils.ops import get_sequence_lengths
from models import (
    build_single_program_encoder,
    build_attention1_decoder
)


def analyze_model(option, use_basic_dataset):
    # BASEDIR = os.path.dirname(os.path.realpath(__file__))
    BASEDIR = 'experiments/RVAE_attention'
    sequence_cap = 56 if use_basic_dataset else 130
    TOKEN_EMB_SIZE = 54
    NUMBER_OF_EXAMPLES = 1000

    if option.startswith('attention1'):
        z_size = int(option.split('_')[-1])
        directory = '{}{}'.format(
            'basic_' if use_basic_dataset else '',
            option
        )
        input_sequence_t = tf.placeholder(
            shape=[1, sequence_cap, TOKEN_EMB_SIZE],
            dtype=tf.float32,
            name='input_sequence'
        )
        sequence_lengths_t = get_sequence_lengths(
            tf.cast(input_sequence_t, tf.int32)
        )
        mus_and_log_sigs = build_single_program_encoder(
            input_sequence_t,
            sequence_lengths_t,
            z_size
        )
        z = mus_and_log_sigs[:, :z_size]
        decoder_input = tf.placeholder(
            shape=[1, z_size],
            dtype=tf.float32,
            name='decoder_input'
        )
        decoder_output, attention_weights_t = build_attention1_decoder(
            decoder_input,
            sequence_lengths_t,
            sequence_cap,
            TOKEN_EMB_SIZE
        )
        token_probs_t = tf.nn.softmax(decoder_output)

        print('z_size={}'.format(z_size))
    else:
        exit('invalid option')

    huzzer_kwargs = BASIC_DATASET_ARGS if use_basic_dataset else {}

    print('Setting up data pipeline...')
    dataset = one_hot_token_dataset(
        batch_size=1,
        number_of_batches=1000,
        cache_path='{}model_analysis_attention'.format(
            'basic_' if use_basic_dataset else ''
        ),
        length=sequence_cap,
        huzzer_kwargs=huzzer_kwargs
    )

    def get_input():
        return np.squeeze(dataset()[0], axis=0).astype(np.float32)

    path = join(BASEDIR, directory)

    saver = tf.train.Saver()

    examples = [get_input() for i in range(NUMBER_OF_EXAMPLES)]

    sess = tf.Session()
    print('Restoring variables...')
    saver.restore(
        sess, tf.train.latest_checkpoint(path, 'checkpoint.txt')
    )
    examples_dir = join(BASEDIR, ('basic_' if use_basic_dataset else '') + option + '_examples')
    mkdir_p(examples_dir)

    # Autoencode_bit
    autoencoded_examples_path = join(examples_dir, 'autoencoded')
    message = 'Autencoding {} examples'.format(
        len(examples),
    )
    for i, input_sequence in enumerate(tqdm(
        examples, desc=message, total=len(examples)
    )):
        dir_for_example = join(autoencoded_examples_path, str(i))
        mkdir_p(dir_for_example)

        z_mus, length = sess.run([z, sequence_lengths_t], feed_dict={
            input_sequence_t: np.expand_dims(input_sequence, 0),
        })

        token_probs, attention_weights = sess.run(
            [token_probs_t, attention_weights_t],
            feed_dict={
                decoder_input: z_mus
            }
        )
        length = np.squeeze(length)
        token_probs = np.squeeze(token_probs)[:length]
        attention_weights = attention_weights[:length]
        input_sequence = input_sequence[:length]

        input_code = example_to_code(input_sequence)
        output_code = example_to_code(token_probs)
        visualize_attention_weights(output_code, attention_weights, join(dir_for_example, 'attention_weights'))

        input_code = example_to_code(input_sequence)
        write_to_file(join(dir_for_example, 'input.hs'), input_code)
        imsave(join(dir_for_example, 'input.png'), input_sequence.T)
        imsave(join(dir_for_example, 'z.png'), z_mus.reshape((z_mus.size // 32, 32)))

        imsave(join(dir_for_example, 'decoder_output.png'), token_probs.T)
        write_to_file(join(dir_for_example, 'autoencoded_code.hs'), output_code)

    # generate_bit
    generated_examples_path = join(examples_dir, 'generated')
    message = 'Generating {} examples'.format(
        len(examples),
    )
    for i in tqdm(
        range(NUMBER_OF_EXAMPLES), desc=message, total=len(examples)
    ):
        dir_for_example = join(generated_examples_path, str(i))
        mkdir_p(dir_for_example)

        z_gen = np.random.normal(0, 0.1, z_size)
        imsave(join(dir_for_example, 'z.png'), z_gen.reshape((z_gen.size // 32, 32)))
        token_probs, attention_weights = sess.run(
            [token_probs_t, attention_weights_t],
            feed_dict={
                decoder_input: np.expand_dims(z_gen, 0)
            }
        )
        token_probs = np.squeeze(token_probs)
        tokens = np.argmax(token_probs, axis=-1)
        end_of_code = np.argmax(tokens == 0) or sequence_cap
        token_probs = token_probs[:end_of_code]
        attention_weights = attention_weights[:end_of_code]
        output_code = example_to_code(token_probs)

        visualize_attention_weights(output_code, attention_weights, join(dir_for_example, 'attention_weights'))

        imsave(join(dir_for_example, 'decoder_output.png'), token_probs.T)
        write_to_file(join(dir_for_example, 'generated_code.hs'), example_to_code(token_probs))


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


def visualize_attention_weights(output_code, attention_weights, path):
    output_code = output_code.replace('\n', '\\n')
    tokens = output_code.split(' ')
    tokens[-1] = '<end>'
    look_behinds = ['z'] + tokens[:-1]

    attention_weights = [x[0] for x in attention_weights]

    padded_weights = []
    for attention_weighting in attention_weights:
        zero_padding = np.zeros(len(attention_weights) - len(attention_weighting))
        padded = np.concatenate((attention_weighting, zero_padding))
        padded_weights += [padded]

    padded_weights = np.stack(padded_weights)

    matplotlib.rc('xtick', labelsize=6)
    matplotlib.rc('ytick', labelsize=6)
    figure_width = (len(attention_weights) / 50) * 6
    plt.figure(figsize=(figure_width, figure_width))
    plt.pcolormesh(padded_weights.T)
    plt.xticks(
        np.arange(len(attention_weights)) + 0.5,
        tokens,
        rotation='vertical',
        fontname='Monospace'
    )
    plt.yticks(
        np.arange(len(attention_weights)) + 0.5,
        look_behinds,
        fontname='Monospace'
    )
    plt.savefig(
        path + '.pdf',
        format='pdf',
        bbox_inches='tight'
    )
    plt.clf()


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
    args = docopt(__doc__, version='N/A')
    option = args.get('<option>')
    use_basic_dataset = args.get('--basic')
    analyze_model(option, use_basic_dataset)
