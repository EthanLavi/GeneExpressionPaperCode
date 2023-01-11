# Import
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
from typing import Tuple

repetitions = 6  # A value that indicates how many times to run the experiment
batch_size = 40  # Value that indicates the batch_size before a parameter update
validation_split = 0  # Value that indicates the validation split to use (for fine-tuning weights)
bias = 0  # An offset to use to stop the model from over-predicting one class
csv_name = input("CSV: ")  # The csv to use
y_col = ['SURVIVAL', 'PROGRESSION', 'RECURRENCE'][
    int(input("Predicting: (1-3)\n1:SURVIVAL\n2:PROGRESSION\n3:RECURRENCE\n")) - 1]  # Value to predict


# Helper Functions
def get_factors(prediction, val) -> Tuple[int, int, int, int]:
    """Function to get True Positive, True Negative, False Positive, False Negative"""
    prediction = np.round(prediction + bias)
    index = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for out in prediction:
        if out == val[index]:
            if val[index] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if out == 1:
                fp += 1
            else:
                fn += 1
        index += 1
    return tp, tn, fp, fn


def get_predictive_values(prediction, val) -> Tuple[float, float, float, float, float, float, float]:
    tp, tn, fp, fn = get_factors(prediction, val)
    # Safe divide
    sdiv = lambda num, div: 0 if num == 0 else num / div
    sensitivity = sdiv(tp, tp + fn)
    specificity = sdiv(tn, tn + fp)
    balanced_acc = (sensitivity + specificity) / 2
    ppv = sdiv(tp, tp + fp)
    npv = sdiv(tn, tn + fn)
    try:
        auc = metrics.roc_auc_score(val, prediction)
    except ValueError:
        auc = 0.5
    acc = sdiv(tp + tn, fp + fn + tp + tn)
    return acc, sensitivity, specificity, balanced_acc, ppv, npv, auc


# Create machine learning model instance
def run_model(features_train, y_train, features_test, y_test) -> Tuple[float, float]:
    # Layer Creation
    model = keras.Sequential([
        keras.layers.Dense(19, input_dim=len(features_train.axes[1]), activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    # Compile and train
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        verbose=0,
        patience=2,
        mode='auto',
        restore_best_weights=True)

    # Calculate the weighting of the classes
    df = pd.concat([y_train, y_test])
    ones = sum(df == 1)
    zeros = sum(df == 0)
    zero_weight = ones / (zeros + ones)
    one_weight = zeros / (zeros + ones)

    model.fit(features_train, y_train, class_weight={0: zero_weight, 1: one_weight}, epochs=100, batch_size=batch_size,
              validation_split=validation_split, verbose=0,
              callbacks=[early_stop])

    # Calculate Evaluation
    loss, accuracy = model.evaluate(features_train, y_train)

    # Accuracy Calculations
    print('Accuracy (TRAINING): %.2f' % (accuracy * 100))

    # Test Network
    predictions = model.predict(features_test)
    acc, sens, spec, test_bal_acc, ppv, npv, auc = get_predictive_values(predictions, y_test.tolist())

    # Predictive values on train of network
    predictions = model.predict(features_train)
    acc, sens, spec, train_bal_acc, ppv, npv, auc = get_predictive_values(predictions, y_train.tolist())
    return train_bal_acc, test_bal_acc


# Stop execution if not main
if __name__ == "__main__":
    # A list of the columns to use when running experiments
    all_columns = ['XRCC6', 'XRCC5', 'XRCC7', 'LIG4', 'LIG3', 'LIG1', 'XRCC4', 'NHEJ1', 'XRCC1', 'DCLRE1C', 'TP53BP1',
                   'BRCA1', 'BRCA2', 'EXO1', 'EXD2', 'POLM', 'POLL', 'POLQ', 'RAD50', 'MRE11', 'NBN', 'TDP1', 'RBBP8',
                   'CTBP1', 'TP53', 'H2AX', 'ERCC1', 'ERCC4', 'RPA1', 'MSH2', 'MSH3', 'RAD1', 'MSH6', 'PMS1', 'MLH1', 'MLH3']
    columns = input("Columns separated by column (enter for all genes):\n").replace(" ", "").split(",")
    if columns == ['']:
        columns = all_columns

    # Import Train Data
    data_csv = pd.read_csv(csv_name, sep=",")
    data_csv.head()
    # Remove old row numbers
    data_csv = data_csv.drop("Unnamed: 0", axis=1)

    train_i = range(0, math.floor(len(data_csv) * 0.7))

    # Make train and test
    train = data_csv.loc[train_i]
    test = data_csv.drop(train_i, axis=0)

    train_features = train[columns]
    train_y = train[y_col]

    test_features = test[columns]
    test_y = test[y_col]

    result = ["X,Train Acc,Test Acc\n"]

    for x in range(6, len(train_y), math.floor(len(train_y) / 25)):
        # For different sizes of training
        train_cut = train_features.head(x)
        train_y_cut = train_y.head(x)

        # Multiple experimental repetitions
        bal_acc_train_arr = []
        bal_acc_test_arr = []
        for rep in range(repetitions):
            bal_acc_train, bal_acc_test = run_model(features_train=train_cut, y_train=train_y_cut,
                                                    features_test=test_features, y_test=test_y)
            bal_acc_train_arr.append(bal_acc_train)
            bal_acc_test_arr.append(bal_acc_test)
        result.append(f"{x},{sum(bal_acc_train_arr) / repetitions},{sum(bal_acc_test_arr) / repetitions}\n")

    print("Repetitions: " + str(repetitions))

    output_file = open(input("Output file:\n"), "w")
    output_file.writelines(result)
    output_file.close()

