"""
Train a model on Omniglot.
"""

import random

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import OmniglotModel, CuneiformModel
from supervised_reptile.omniglot import read_dataset, split_dataset, augment_dataset
import supervised_reptile.cuneiform as cuneiform
from supervised_reptile.train import train

DATA_DIR_TRAIN = 'data/omniglot'
DATA_DIR_TEST = 'data/cuneiform'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    #for training
    train_set, test_set = split_dataset(read_dataset(DATA_DIR_TRAIN))
    train_set = list(augment_dataset(train_set))
    test_set = list(test_set)

    #for evaluation
    train_eval, test_eval = split_dataset(cuneiform.read_dataset(DATA_DIR_TEST))
    train_eval = list(augment_dataset(train_eval))
    test_eval = list(test_eval)

    model_train = OmniglotModel(args.classes, **model_kwargs(args))

    model_test = CuneiformModel

    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model_train, train_set, test_set, args.checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        print('Train accuracy: ' + str(evaluate(sess, model_test, train_eval, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model_test, test_eval, **eval_kwargs)))

if __name__ == '__main__':
    main()
