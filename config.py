from models.cnn import TextClassificationCNN

CLASS2NAME = {
    'negative': 0,
    'positive': 1,
}

NAME2MODEL = {
    'CNN': TextClassificationCNN,
}