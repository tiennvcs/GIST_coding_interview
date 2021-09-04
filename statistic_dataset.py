import os
import argparse
from config import CLASS2NAME
from nltk.tokenize import word_tokenize
import numpy as np
import json


def load_data_by_path(path):
    X = []
    labels = []
    for subfoder in os.listdir(path):
        class_id = CLASS2NAME[subfoder]
        for file_path in os.listdir(os.path.join(path, subfoder)):
            
            # Read sample from file
            with open(os.path.join(path, subfoder, file_path), 'r') as f:
                reading_text = f.read()
            
            # Tokenize paragraph to list of words
            words = word_tokenize(reading_text) 
            # Remove punctuation
            # words = [word for word in words if word.isalpha()]

            X.append(words)
            labels.append(class_id)
    
    return np.array(X), np.array(labels)


def statistic(data):
    X, label = data

    words = []
    for list_word in X:
        words += list_word
    length_lst = [len(list_word) for list_word in X]
    
    # total number of unique words in T
    unique_words = set(words)
    number_unique = len(unique_words)
    print("\t+ The number of unique words: {}".format(number_unique))

    # total number of training examples
    N = X.shape[0]
    print("\t+ The number of training examples: {}".format(N))

    # ratio of positive examples to negative examples
    positives = np.where(label == CLASS2NAME['positive'])[0]
    negatives = np.where(label == CLASS2NAME['negative'])[0]
    ratio_pn = len(positives)/len(negatives)
    print("\t+ Ratio of positive examples to negative examples: {}".format(ratio_pn))

    # average length of document
    average_len = np.mean(length_lst)
    print("\t+ The average length of document: {}".format(average_len))

    # maximum length of document
    max_length = int(np.max(length_lst))
    print("\t+ The maximum length of document is {}".format(max_length))

    return {
        'number_unique': number_unique,
        'number_training_examples': N,
        'n_positives': len(positives),
        'n_negatives': len(negatives),
        'ratio_pn': ratio_pn,
        'average_length': average_len,
        'max_length': max_length,
    }


def main(args):

    # Load data from path
    print("[info] Loading data ...")
    data = load_data_by_path(path=args['input_dir'])

    # Statistic the dataset
    print("[info] Statistic dataset ...")
    params = statistic(data)
    print(params)

    # Save params to disk
    with open('runs/statistic_output.json', 'w') as f:
        json.dump(params, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Statistic the data set inside a specfic path directory")
    
    parser.add_argument('--input_dir', default='./data/train', 
        help='The path to dataset'
    )

    args = vars(parser.parse_args())
    main(args)