# Simple n-gram (Markov chain) model for character-based text generation.
#
# Only tested with Python 3.6+
#
# Eli Bendersky (https://eli.thegreenplace.net)
# This code is in the public domain
from __future__ import print_function

from collections import defaultdict, Counter
import random
import sys


# This is the length of the "state" the current character is predicted from.
# For Markov chains with memory, this is the "order" of the chain. For n-grams,
# the n is STATE_LEN+1 since it includes the predicted character as well.
STATE_LEN = 4


def weighted_from_counter(c):
    total = sum(c.values())
    idx = random.randrange(total)
    for elem, count in c.most_common():
        idx -= count
        if idx < 0:
            return elem


def main():
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        data = f.read()

    states = defaultdict(Counter)

    print('Learning model...')
    for i in range(len(data) - STATE_LEN - 1):
        state = data[i:i + STATE_LEN]
        next = data[i + STATE_LEN]
        states[state][next] += 1

    print('Model has {0} states'.format(len(states)))
    j = 0
    for k, v in states.items():
        print(k, v)
        if j > 9:
            break
        j += 1

    print('Sampling...')
    state = random.choice(list(states))
    sys.stdout.write(state)
    for i in range(200):
        nextc = weighted_from_counter(states[state])
        sys.stdout.write(nextc)
        state = state[1:] + nextc
    print()


if __name__ == '__main__':
    main()
