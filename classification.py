#!/usr/bin/python
"""Train a classification and regression model to predict CSE effectiveness."""


import argparse
from typing import List

from pandas import DataFrame, get_dummies
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from jitml import get_individual_cse_perf
from train import validate_core_root

# At what perf_score would we want to select a CSE?  We don't want to select any CSE which
# is 0.0, as we are creating new temporaries for no value.
# I've arbitrarily selected this minimum perf_score improvement for a "successful" CSE.
CSE_SUCCESS_THRESHOLD = -5.0

# We will only retain features which have a correlation of at least 1% with the change in
# perf_score.  This is to reduce the number of features we have to consider.  Selecting a
# value as high as 15% would still be reasonable, but since there are so few features we
# can afford to allow a lower threshold.
CORRELATION_THRESHOLD = 0.01

# We will use a 90/10 train/test split.
TRAIN_TEST_SPLIT = 0.1

def parse_args():
    """usage:  classification.py [-h] [--core_root CORE_ROOT] [--parallel n] mch"""
    parser = argparse.ArgumentParser()
    parser.add_argument("mch", help="The mch file of functions to train on.")
    parser.add_argument("--core_root", default=None, help="The coreclr root directory.")
    parser.add_argument("--layer-density", type=str, default="32,32", help="The number of neurons in each layer.")
    parser.add_argument("--optimizer", type=str, default="adam", help="The optimizer to use.")
    parser.add_argument("--loss", type=str, default="binary_crossentropy", help="The loss function to use.")
    parser.add_argument("--epochs", type=int, default=100, help="The number of epochs to train for.")
    parser.add_argument("--kind", type=str, default="classification",
                        help="The kind of model to train (classifcation | regression).")

    args = parser.parse_args()
    args.core_root = validate_core_root(args.core_root)
    args.layer_density = [int(x) for x in args.layer_density.split(",")]
    return args

def sanitize_data(df : DataFrame, is_classification) -> DataFrame:
    """Sanitize the data for training based on the kind of model."""

    # see notebooks/00_random_forest.ipynb for more information on the approach here
    result = get_dummies(df, columns=['type'])
    if is_classification:
        result['target'] = result['cse_score'] - result['no_cse_score'] < CSE_SUCCESS_THRESHOLD
    else:
        result['target'] = result['cse_score'] - result['no_cse_score']

    to_drop = ['method', 'cse_index', 'cse_score', 'no_cse_score', 'heuristic_score', 'heuristic_selected',
               'index', 'applied', 'viable', 'def_count', 'type_JitType.FLOAT', 'type_JitType.SIMD']

    result.drop(columns=to_drop, inplace=True)
    return result

def build_model(input_size : int, layer_density : List[int], optimizer : str, loss : str, is_classification : bool):
    """Build a model of the requested structure."""

    l = []
    for size in layer_density:
        if layers:
            l.append(layers.Dense(size, activation='relu'))
        else:
            l.append(layers.Dense(size, input_shape=(input_size,), activation='relu'))

    # either a binary classification or a regression
    if is_classification:
        l.append(layers.Dense(1, activation='sigmoid'))
    else:
        l.append(layers.Dense(1))

    model = tf.keras.Sequential(l)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


def main(args):
    """Main entry point."""

    # get the data
    df = get_individual_cse_perf(args.mch, args.core_root)
    df = sanitize_data(df, args.kind == "classification")

    x, y = df.drop(columns=['target']), df['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TRAIN_TEST_SPLIT)

    print(f"Training on {len(x_train)} samples, testing on {len(x_test)} samples.")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # build and fit the model
    model = build_model(len(df.columns) - 1, args.layer_density, args.optimizer, args.loss, args.kind == "classification")
    model.fit(x_train, y_train, epochs=args.epochs, validation_data=(x_test, y_test))

    train_loss, train_acc = model.evaluate(x_train, y_train)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")



if __name__ == '__main__':
    main(parse_args())
