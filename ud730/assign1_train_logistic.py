from __future__ import print_function
from six.moves import cPickle as pickle
from sklearn.linear_model import LogisticRegression

from utils import show_image, shuffle_data_and_labels, Timer


PICKLE_DATA = 'notMNIST.pickle'

# Load data from the pickle
data = pickle.load(open(PICKLE_DATA, 'r'))
for key in data:
    print(key, ':', data[key].shape)


def get_data_and_labels(dataset, labelset, nmax=None, shuffle=False):
    """Createa a dataset and labelset from pickled inputs.

    Reshapes the data from samples of 2D images (N, 28, 28) to linearized
    samples (N, 784). Also cuts a subset of the data/label-set when nmax is
    set. shuffle lets us reshuffle the set before cutting.
    """
    if shuffle:
        d, l = shuffle_data_and_labels(dataset, labelset)
    else:
        d, l = dataset, labelset

    assert l.ndim == 1 and d.shape[0] == l.shape[0]
    if nmax is None:
        nmax = l.shape[0]

    # Assuming data comes in with shape (nsamples, M, N); we linearize each
    # sample to transform it to (nsamples, M*N).
    assert d.ndim == 3
    d, l = d[:nmax, :].reshape(nmax, -1), l[:nmax]
    return d, l


NTRAIN = 500

with Timer('preprocess'):
    traindata, trainlabels = get_data_and_labels(data['train_dataset'],
                                                 data['train_labels'],
                                                 nmax=NTRAIN,
                                                 shuffle=False)

#print(d50_data.shape, d50_labels.shape)
#img7 = d50_data[17].reshape(28, 28)
#print(chr(ord('A') + d50_labels[17]))
#show_image(img7)

reg = LogisticRegression(C=1)

with Timer('fit'):
    reg.fit(traindata, trainlabels)

testdata, testlabels = get_data_and_labels(data['test_dataset'],
                                           data['test_labels'])

print('Test data shape:', testdata.shape)
print('Test labels shape:', testlabels.shape)

with Timer('score'):
    print(reg.score(testdata, testlabels))
