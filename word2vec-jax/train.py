import numpy as np
import pickle


def generate_train_vectors(train_data, vocab, window_size=4, batch_size=128):
    """Generates training vectors from a list of words and vocabulary.
    
    Generates (target_batch, context_batch) pairs, where target_batch is a
    (batch_size,) array of target word IDs and context_batch is a (batch_size,)
    array of corresponding contexts; each context is an (2*window_size,) array
    of word IDs.

    Stops when it runs out of data. The last batch may have fewer elements than
    batch_size.
    """
    target_batch = []
    context_batch = []

    for i in range(len(train_data)):
        if i + 2 * window_size >= len(train_data):
            break

        # 'i' is the index of the leftmost word in the context.
        target_word = train_data[i + window_size]
        left_context = train_data[i : i + window_size]
        right_context = train_data[i + window_size + 1 : i + 2 * window_size + 1]
        context_words = left_context + right_context
        
        if target_word not in vocab:
            continue
        target_batch.append(vocab.get(target_word, '<unk>'))
        context_batch.append([vocab.get(word, '<unk>') for word in context_words])

        if len(target_batch) == batch_size:
            yield np.array(target_batch), np.array(context_batch)
            target_batch = []
            context_batch = []
    

if __name__ == "__main__":
    # Load train-data from pickle file
    with open("train-data.pickle", "rb") as file:
        data = pickle.load(file)
        train_data = data["train_data"]
        vocab = data["vocab"]
    print("Number of words in train data:", len(train_data))
    print("Vocabulary size:", len(vocab))
    print(
        "First 10 words and their IDs:",
        [(word, vocab[word]) for word in train_data[:10]],
    )

    cnt = 0
    for target_batch, context_batch in generate_train_vectors(train_data, vocab, batch_size=4):
        cnt += 1
        # print("Target batch:", target_batch)
        # print("Context batch:", context_batch)
        # print()

    print("Number of batches:", cnt)

