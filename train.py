from GIST_coding_interview.data_processing import processing
from random import random

import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from utils import split_dataset
from config import NAME2MODEL
from data_processing import data_processing


def train(X, y, model, test_size, epoch, batch_size, device):
    pass
    


def main(args):
    # Load dataset from disk
    data = data_processing(input_dir=args['data_dir'], num_words=args['num_words'])
    # Initialize model





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training text classifier')
    
    parser.add_argument('--data_dir', default='./data/train/',
        help='The training data directory path'
    )

    parser.add_argument('--model', default='CNN',
        help='The using model'
    )
    parser.add_argument('--pretrain_embedding', default=None,
        help='The pretrain embedding',
    )

    parser.add_argument('--test_size', default=0.2, type=float,
        help="The test size for spliting dataset",
    )

    parser.add_argument('--device', default='cuda:0',
        help='The selected gpu device'
    )

    parser.add_argument('--epochs', default=100, type=int,
        help='The number of epochs in training progress'
    )

    parser.add_argument('--batch_size', default=32, type=int,
        help='The batch size using in training progress'
    )

    parser.add_argument('--word_dir', default='./runs/train/cnn_train/',
        help='The working directory to store model and training log'
    )

    args = vars(parser.parse_args())
    main(args)
