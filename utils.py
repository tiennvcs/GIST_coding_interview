import json
import numpy as np
from sklearn.model_selection import train_test_split
import os
from matplotlib import pyplot as plt
from statistic_dataset import load_data_by_path
from keras.preprocessing.text import Tokenizer
import tensorflow as tf


def count_word_frequency(X):
    words = []
    for list_word in X:
        words += list_word
    word_frequency = {}
    for word in words:
        if not word in word_frequency.keys():
            word_frequency[word] = 1
        else:
            word_frequency[word] += 1
    return word_frequency


def get_top_frequency(frequency_dict, k):

    words = []
    frequencies = []
    for word, count in frequency_dict.items():
        words.append(word)
        frequencies.append(count)
    sorted_indices = np.argsort(-np.array(frequencies))[:k]
    words = np.array(words, dtype=object)
    return words[sorted_indices]


def get_most_words(X, number_words):

    # Count frequency of each word
    word_frequency = count_word_frequency(X=X)
    
    # Get top k most appearing words
    most_words =  get_top_frequency(frequency_dict=word_frequency, k=number_words)
    return most_words


def load_word_embedding(path):
    """
        Return dictionary of word_embeddings loading from disk
        {
            word_1: [] -> np.ndarray() , shape = (100, )
            word_2: [] -> np.ndarray()
            ...
        }
    """
    word_embeddings = {}

    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()[1:]
    for line in data:
        line = line.split()
        word = line[0]
        embedding = np.array([float(v) for v in line[1:]])
        word_embeddings[word] = embedding
    return word_embeddings


def build_data(X, vocab, word_embeddings, dim):
    """
        Return data have shape: [len(X), len(vocab), len(word_embeddings[0])]
    """
    data = []
    zero_feature_vector = np.zeros(shape=dim)

    for list_word in X:
        feature_vectors = []
        for word in vocab:
            # if word is not in word_embeddings or is not in list_word then we ignore it
            if (not word in list_word) or (not word in word_embeddings.keys()): 
                feature_vectors.append(zero_feature_vector)
            else:
                feature_vectors.append(word_embeddings[word])
        data.append(np.array(feature_vectors, dtype=object))
    data = np.array(data)
    return data

  

def split_dataset(X, y, ratio):
    """
        Split dataset into training/testing dataset for training stage
    """
    X_train, y_train, X_test, y_test = train_test_split(
        X, y, test_size=ratio, random_state=2908,
    )
    return X_train, y_train, X_test, y_test


def rename_dataset(subset):
    all_files = os.listdir(os.path.join(subset, 'negative/'))
    for name_file in tqdm(all_files):
        src = os.path.join(subset, 'negative', name_file)
        dst = os.path.join(subset, 'negative', name_file) + '.txt'
        os.rename(src, dst)


def plot_history(history, word_dir, model_name, n_epochs):

    # Saving accuracy training progress
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('{} accuracy'.format(model_name))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    saving_path = os.path.join(word_dir, 'accuracy_log.png')
    plt.savefig(saving_path)
    plt.clf()

    # Saving loss accuracy in training progress
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('{} objective'.format(model_name))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    saving_path = os.path.join(word_dir, 'loss_log.png')
    plt.savefig(saving_path)


def create_embedding_matrix(path, word_index, num_feature, num_dim):
    hits = 0
    misses = 0
    pretrain_embedding = load_word_embedding(path=path)
    embedding_matrix = np.zeros((num_feature, num_dim))
    for word, i in word_index.items():
        embedding_vector = pretrain_embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix


def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.1)