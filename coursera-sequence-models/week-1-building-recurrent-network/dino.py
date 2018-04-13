import numpy as np
from rnn_provided import *
import random

data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
print(ix_to_char)


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients
                 "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number,
                and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
    dWaa, dWax, dWya, db, dby = (gradients['dWaa'], gradients['dWax'],
                                 gradients['dWya'], gradients['db'],
                                 gradients['dby'])
   
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, a_min=-10, a_max=10, out=gradient)
    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients


def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability
    distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the
                  parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the
               sampled characters.
    """
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = (parameters['Waa'], parameters['Wax'],
                            parameters['Wya'], parameters['by'],
                            parameters['b'])
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    n_x = Wax.shape[1]
    
    ### START CODE HERE ###
    # Step 1: Create the one-hot vector x for the first character (initializing
    # the sequence generation).
    x = np.zeros((n_x, 1))
    # Step 1': Initialize a_prev as zeros
    a_prev = np.zeros((n_a, 1))
    
    # Create an empty list of indices, this is the list which will contain the
    # list of indices of the characters to generate.
    indices = []
    
    # Idx is a flag to detect a newline character, we initialize it to -1.
    idx = -1 
    
    # Loop over time-steps t. At each time-step, sample a character from a
    # probability distribution and append its index to "indices". We'll stop if
    # we reach 50 characters (which should be very unlikely with a well trained
    # model), which helps debugging and prevents entering an infinite loop. 
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        
        # for grading purposes
        np.random.seed(counter+seed) 
        
        # Step 3: Sample the index of a character within the vocabulary from the
        # probability distribution y
        idx = np.random.choice(np.arange(vocab_size), p=y.ravel())

        # Append the index to "indices"
        indices.append(idx)
        
        # Step 4: Overwrite the input character as the one corresponding to the sampled index.
        x = np.zeros((n_x, 1))
        x[idx, 0] = 1
        
        # Update "a_prev" to be "a"
        a_prev = a
        
        # for grading purposes
        seed += 1
        counter +=1

    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices


if __name__ == '__main__':
    np.random.seed(2)
    _, n_a = 20, 100
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

    indices = sample(parameters, char_to_ix, 0)
    print("Sampling:")
    print("list of sampled indices:", indices)
    print("list of sampled characters:", [ix_to_char[i] for i in indices])
