import time
import argparse
from statistic_dataset import load_data_by_path
from utils import get_most_words, load_word_embedding, build_data


def data_processing(input_dir, num_words, word_embedding):
    
    # Load data from directory
    X, y = load_data_by_path(input_dir)

    # Get vocabulary
    print("[info] Getting list of the most appearing words ...")
    most_words = get_most_words(X=X, number_words=num_words)

    # Load word_embedddings
    print("[info] Loading word embeddings ...")
    word_embeddings = load_word_embedding(path=word_embedding)

    # Build feature embedding base on most_words and word_embeddings
    print("[info] Building the dataset base on word embeddings and most_words...")
    data = build_data(X=X, vocab=most_words, word_embeddings=word_embeddings, dim=100)

    print("[info] Finish ! ")
    print(data.shape)
    return (data, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing dataset and store to disk")
    
    parser.add_argument('--input_dir', default='./data/train', 
        help='The path to dataset'
    )
    
    parser.add_argument('--num_words', default=10000,
        help='The number of most frequency word using'
    )

    parser.add_argument('--word_embedding', default='./data/all.review.vec.txt',
        help='Word embedding path'
    )
    
    parser.add_argument('--output_path', default='runs/embeddings/dataset_embedding.json', 
        help='Output file for embedding dataset'
    )

    args = vars(parser.parse_args())

    start_time = time.time()
    data_processing(args['input_dir'], args['num_words'], args['word_embedding'])
    print("--- %s seconds ---" % (time.time() - start_time))