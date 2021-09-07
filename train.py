import os
import time
import argparse
import tensorflow as tf
from prepare_data import build_dataset
from config import NAME2OPTIMIZER, NAME2MODEL
from utils import create_embedding_matrix, scheduler, plot_history


def train(args):
    
    if not os.path.exists(args['work_dir']):
        os.mkdir(args['work_dir'])


    # Build dataset
    print("[info] Loading dataset from {}...".format(args['data_dir']))
    train_ds, val_ds, test_ds, word_index = build_dataset(input_dir=args['data_dir'], 
                                                        split_ratio=args['test_size'],
                                                        batch_size=args['batch_size'],
                                                        num_words=args['num_words'],
                                                        num_dim=args['feature_dim'])
    
    embedding_matrix = None
    use_pretrain_embedding = False
    # Get the embedding matrix if provided
    if args['pretrain_embedding']:
        print("[info] Loading pretraining embedding ...")
        use_pretrain_embedding = True
        embedding_matrix = create_embedding_matrix(
            path=args['pretrain_embedding'],
            word_index=word_index,
            num_feature=args['num_words'],
            num_dim=args['feature_dim']
        )

    print("[info] Creating model {}...".format(args['model']))
    # Create model
    model = NAME2MODEL[args['model'].lower()](use_pretrain_embedding=use_pretrain_embedding,
                                    embedding_matrix=embedding_matrix)
    model.summary()
    # input("Press [Enter] to start TRAINING .... :)")
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                optimizer=NAME2OPTIMIZER[args['optimizer']](learning_rate=args['learning_rate']),
                metrics=["accuracy"]
    )   

    # Define callbacks
    checkpoint_filepath = args['work_dir']
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=True,
                    monitor='val_accuracy',
                    save_best_only=True
    )
    scheduler_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Training model
    print("[info] Training in {} epochs with {} batch size...".format(args['epochs'], args['batch_size']))
    start_time = time.time()
    history = model.fit(train_ds, validation_data=val_ds, 
                        epochs=args['epochs'], batch_size=args['batch_size'],
                        callbacks=[checkpoint, scheduler_lr])
    print("--> Training time: {}".format(time.time() - start_time))

    # Evaluate model
    model.load_weights(checkpoint_filepath)
    model.evaluate(test_ds)

    # Saving the training accuracy 
    plot_history(history=history, word_dir=args['work_dir'], 
                model_name=args['model'], n_epochs=args['epochs'],
    )    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training text classifier')
    
    parser.add_argument('--data_dir', default='./data',
        help='The training data directory path'
    )

    parser.add_argument('--model', default='CNN', choices=['cnn', 'rnn', 'CNN','RNN'],
        help='The using model'
    )
    parser.add_argument('--pretrain_embedding', default=None,
        help='The pretrain embedding',
    )

    parser.add_argument('--num_words', default=10000, type=int,
        help='Maximum number of words',
    )

    parser.add_argument('--feature_dim', default=100, type=int,
        help='Maximum number of feature dim',
    )

    parser.add_argument('--test_size', default=0.2, type=float,
        help="The test size for spliting dataset",
    )

    parser.add_argument('--epochs', default=10, type=int,
        help='The number of epochs in training progress'
    )

    parser.add_argument('--batch_size', default=64, type=int,
        help='The batch size using in training progress'
    )

    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'rmsprop'],
        help='The optimization algorithm using in training'
    )

    parser.add_argument('--learning_rate', default=1.0e-3, type=float,
        help='The learning rate of model'
    )
    
    parser.add_argument('--work_dir', default='./runs/train/cnn_train/',
        help='The working directory to store model and training log'
    )

    args = vars(parser.parse_args())
    print(args)
    train(args)
