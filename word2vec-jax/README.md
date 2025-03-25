This is a reproduction of the classical word2vec embedding model using JAX.

There are several steps to get a trained model we can use to query for
embeddings.

# 1. Download the training set

Any training set of text will do, as long as it's properly cleaned up. For this
project I'm using the same set used by the
[original word2vec project](https://code.google.com/archive/p/word2vec/).
The `download-dataset.sh` script should be run - it downloads a large (100 MB)
text file named `text8`. This is just a lot of concatenated English text,
space separated, with no punctuation.

If you'd like to use a different dataset (such as Wikipedia dumps), make sure
the input file has the same format as `text8`.

# 2. Prepare data for training

To prepare the data for training, we want to
sub-sample the input (reducing the frequency of very common words), and compile
a vocabulary of the N most common words (20,000 by default).

After step 1 is completed, run:

    uv run make-train-data.py

This creates a file named `train-data.pickle`, which can be directly loaded
by subsequent steps. See the script top-level-comment for details on what
this file contains.

# 3. Train the model

To train the model, run:

    uv run train.py

This reads `train-data.pickle` and trains a CBOW word2ver model using JAX. The
training process runs multiple epochs over the data set in shuffled batches,
and saves a checkpoint every epoch. The checkpoint contains a dict with two
arrays representing the model's layers.

Of particular interest is the `projection` array, shaped (V, D) where V
is our vocabulary size and D our embedding model depth. This is the embedding
table, mapping word index to the word's learned embedding.

# 4. Query embeddings for similar words and analogies

Once the model has trained sufficiently for 15-25 epochs (the loss should go down),
the latest checkpoint can be used to query the embeddings for word similarities
and analogies. Here are some examples:

    # Find the most similar words to "paris"
    uv run similar-words.py -word paris -checkpoint checkpoint.pickle -traindata train-data.pickle

    # Find the best analogies for "berlin is to germany as tokyo is to ??"
    uv run similar-words.py -analogy berlin,germany,tokyo -checkpoint checkpoint.pickle -traindata train-data.pickle


