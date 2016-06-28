"""Softmax."""

scores = [1.0, 2.0, 3.0]

import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exps = np.exp(x)
    sumcols = np.sum(exps, axis=0)
    return exps / sumcols



print(softmax(scores))

# Plot softmax curves
#import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])


scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]]) * 0.1

print(scores)
print(softmax(scores))

#plt.plot(x, softmax(scores).T, linewidth=2)
#plt.show()
