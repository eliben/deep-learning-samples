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
