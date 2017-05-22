"""Mean Levenshtein Distance Calculator.

Usage:
  mean_levenshtein_dist.py --original-source=<orig-filename> --computed-source=<comp-filename> <example_dir>
  mean_levenshtein_dist.py -h | --help
  mean_levenshtein_dist.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  -o --original-source=<orig-filename> Name of the original haskell code.
  -c --computed-source=<comp-filename> Name of the computed code.
"""

from docopt import docopt
import Levenshtein as lv
from os import path
import os
from huzzer.tokenizing import tokenize
from tqdm import tqdm


def main(example_dir, original_filename, computed_filename):

    # path_for_example = '0'

    directories = [
        path.join(example_dir, f) for f in os.listdir(example_dir)
        if path.isdir(path.join(example_dir, f))
    ]

    total_normed_levenshtein_distance = 0
    i = 0

    for directory in tqdm(directories):
        if (
            not path.isfile(path.join(directory, original_filename)) or
            not path.isfile(path.join(directory, computed_filename))
        ):
            continue

        i += 1
        with open(
            path.join(directory, original_filename)
        ) as f:
            original_code = f.read()
        with open(
            path.join(directory, computed_filename)
        ) as f:
            computed_code = f.read()

        original_tokens = text_to_token_ids(original_code)
        computed_tokens = text_to_token_ids(computed_code)

        original_tokens_string = ''.join([str(x) for x in original_tokens])
        computed_tokens_string = ''.join([str(x) for x in computed_tokens])

        levenshtein_distance = lv.distance(
            original_tokens_string, computed_tokens_string
        )
        normed_levenshtein_distance = levenshtein_distance / len(original_tokens_string)
        total_normed_levenshtein_distance += normed_levenshtein_distance
        with open(path.join(directory, 'levenshtein_score.txt'), 'w') as f:
            score_text = 'levenshtein_score {}\nlength {}\n levenshtein_distance {}'.format(
                normed_levenshtein_distance,
                len(original_tokens_string),
                levenshtein_distance
            )
            f.write(score_text)

    average_normed_score = total_normed_levenshtein_distance / i
    with open(path.join(example_dir, 'levenshtein_score.txt'), 'w') as f:
        f.write(str(average_normed_score))

    print('mean levenshtein_score for example_dir:\n{}'.format(average_normed_score))


def text_to_token_ids(code):
    return [x.type for x in tokenize(code) if x.channel == 0]



if __name__ == '__main__':
    args = docopt(__doc__, version='0.0.1')
    example_dir = args.get('<example_dir>')
    original_filename = args.get('--original-source')
    computed_filename = args.get('--computed-source')
    main(example_dir, original_filename, computed_filename)
