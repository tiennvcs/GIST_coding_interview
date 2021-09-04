from tensorflow.keras import layers
import tensorflow as tf
from utils import create_embedding_matrix


def TextClassificationCNN(num_words=10000, input_length=100, output_dim=100, use_pretrain_embedding=False, embedding_matrix=None):

    inputs = tf.keras.Input(shape=(None,), dtype="int64")
    if not use_pretrain_embedding:
        x= tf.keras.layers.Embedding(input_dim=num_words, 
                                    output_dim=output_dim, 
                                    input_length=input_length)(inputs)
    else:
        x = layers.Embedding(input_dim=embedding_matrix.shape[0], 
                            input_length=embedding_matrix.shape[1], 
                            output_dim=embedding_matrix.shape[1])(inputs)
    x = layers.Conv1D(100, 3, padding="valid", activation="relu", strides=2)(x)
    x = layers.Dense(100)(x)
    preds = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    model = tf.keras.Model(inputs, preds)
    if use_pretrain_embedding:
        model.layers[1].set_weights([embedding_matrix])
        model.layers[1].trainable = False
    return model
    

if __name__ == '__main__':
    model = TextClassificationCNN()
    model.summary()