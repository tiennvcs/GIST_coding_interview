import tensorflow as tf


CLASS2NAME = {
    'negative': 0,
    'positive': 1,
}


NAME2OPTIMIZER = {
    'rmsprop': tf.keras.optimizers.RMSprop,
    'sgd':  tf.keras.optimizers.SGD,
    'adam': tf.keras.optimizers.Adam,
}