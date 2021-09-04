from tensorflow.keras.layers import TextVectorization
import string
import re
import tensorflow as tf
import os


def build_dataset(input_dir, split_ratio, batch_size, num_words, num_dim):

    # raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    #     os.path.join(input_dir, 'train'),
    #     batch_size=batch_size,
    #     validation_split=split_ratio,
    #     subset="training",
    #     seed=2908,
    # )

    # raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    #     os.path.join(input_dir, 'train'),
    #     batch_size=batch_size,
    #     validation_split=split_ratio,
    #     subset="validation",
    #     seed=2908,
    # )

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        os.path.join(input_dir, 'train'), batch_size=batch_size,
    )

    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        os.path.join(input_dir, 'test'), batch_size=batch_size
    )

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        os.path.join(input_dir, 'test'), batch_size=batch_size
    )

    def mystandardize(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
        return tf.strings.regex_replace(
            stripped_html, "[%s]" % re.escape(string.punctuation), ""
        )

    vectorize_layer = TextVectorization(
        standardize=mystandardize,
        max_tokens=num_words,
        output_mode="int",
        output_sequence_length=num_dim,
    )

    text_ds = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    # Vectorize the data.
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)
    test_ds = test_ds.cache().prefetch(buffer_size=10)

    voc = vectorize_layer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    return train_ds, val_ds, test_ds, word_index


if __name__ == '__main__':
     # Model constants.
    num_words = 10000
    num_dim = 500
    split_ratio = 0.2
    train_ds, val_ds, text_ds = build_dataset(input_dir='./data', 
                                    batch_size=32, split_ratio=split_ratio,
                                    num_words=num_words, 
                                    num_dim=num_dim)