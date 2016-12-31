"""
Generate code from a model.

Usage:
  generate_code_recurrent.py --model=<modelpath> [--temperature=<temp>] [--number=<num>] <path>
  generate_code_recurrent.py -h | --help
  generate_code_recurrent.py --version

Options:
  -h --help                Show this screen.
  --version                Show version.
  -m --model=<modelpath>   Path to model to generate
  -t --temperature=<temp>  Temperature for model generation [default: 0.3]
  -n --number=<num>        Number of times to generate some code [default: 1]
"""
from docopt import docopt
import numpy as np
from keras.models import load_model
from tqdm import trange


def main():
    args = docopt(__doc__, version='0.0.1')

    modelpath = args.get('--model')
    model = load_model(modelpath)
    temperature = float(args.get('--temperature'))

    num_iters = int(args.get('--number'))
    path = args.get('<path>')

    power_of_ten = int(np.log10(num_iters))

    for i in trange(num_iters):
        text = generate(model, temperature)
        filename = path + 'temp{}_{}.txt'.format(temperature, format(i, '0{}d'.format(power_of_ten)))
        with open(filename, 'w') as text_file:
            print(text, file=text_file)  # NOQA


def generate(
    model,
    temperature=0.35,
    seed=None,
    predicate=lambda x: len(x) < 512
):
    # TODO: padding with spaces or nothing
    # TODO: this max len should be gotten from the model
    max_len = 32

    generated = seed
    if seed is not None and len(seed) < max_len:
        raise Exception('Seed text must be at least {} chars long'.format(max_len))

    # if no seed text is specified, randomly select a chunk of text
    else:
        # start_idx = random.randint(0, len(text) - max_len - 1)
        # seed = text[start_idx:start_idx + max_len]
        # TODO: dynamic ssizing here
        # TODO: spacing char is currently space, althogh it should be nothing
        seed = ' ' * max_len
        generated = ''

    sentence = seed

    while predicate(generated):
        # generate the input tensor
        # from the last max_len characters generated so far
        x = np.zeros((1, max_len, 128))
        for t, char in enumerate(sentence):
            x[0, t, char_to_idx(char)] = 1.

        # this produces a probability distribution over characters
        probs = model.predict(x, verbose=0)[0]

        # sample the character to use based on the predicted probabilities
        next_idx = sample(probs, temperature)

        next_char = chr(next_idx)

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated


def sample(probs, temperature):
    """samples an index from a vector of probabilities"""
    probs[probs == 1.0] = 0.999
    # import code
    # code.interact(local=locals())

    # print(np.log(probs))


    a = np.log(probs)/temperature
    a = np.exp(a)/np.sum(np.exp(a))

    # HACK: to keep numpy from whining
    a = a / (a.sum() + 0.001)
    return np.argmax(np.random.multinomial(1, a, 1))


def char_to_idx(c):
    # TODO: this is not really the end behaviour, more change is needed in other code (i.e. arr[arange(l), x]) = 1)
    if ord(c) < 128:
        return ord(c)
    else:
        return 128


if __name__ == '__main__':
    main()
