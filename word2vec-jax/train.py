import pickle

if __name__ == "__main__":
    # Load train-data from pickle file
    with open("train-data.pickle", "rb") as file:
        data = pickle.load(file)
        train_data = data["train_data"]
        vocab = data["vocab"]
    print("Number of words in train data:", len(train_data))
    print("Vocabulary size:", len(vocab))
    print("First 10 words and their IDs:",
          [(word, vocab[word]) for word in train_data[:10]])
    