"""
Train a model on Omniglot.
"""

import random

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import OmniglotModel, CuneiformModel
from supervised_reptile.omniglot import read_dataset, split_dataset
import supervised_reptile.cuneiform as cuneiform
from supervised_reptile.train import train
import os

DATA_DIR = 'data/cuneiform'

def main():
    #use only one gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    #for evaluation
    train_eval, test_eval = split_dataset(cuneiform.read_dataset(DATA_DIR), 5)
    train_eval = list(train_eval)
    test_eval = list(test_eval)

    model_test = CuneiformModel(3, **model_kwargs(args))

    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model_test, train_eval, test_eval, args.checkpoint, **train_kwargs(args))
            print("after training")
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        print('Train accuracy: ' + str(evaluate(sess, model_test, train_eval, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model_test, test_eval, **eval_kwargs)))

if __name__ == '__main__':
    main()
