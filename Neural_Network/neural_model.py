# Import
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List
import time

# Global Variables
model_acc: List[List[float]] = [[] for _ in range(10)]
model_sens: List[List[float]] = [[] for _ in range(10)]
model_spec: List[List[float]] = [[] for _ in range(10)]
model_bal: List[List[float]] = [[] for _ in range(10)]
model_ppv: List[List[float]] = [[] for _ in range(10)]
model_npv: List[List[float]] = [[] for _ in range(10)]

repetitions = 10  # A value that indicates how many times to run the experiment
batch_size = 40  # Value that indicates the batch_size before a parameter update
validation_split = 0.1  # Value that indicates the validation split to use (for fine-tuning weights)
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


def get_predictive_values(prediction, val) -> Tuple[float, float, float, float, float, float]:
    tp, tn, fp, fn = get_factors(prediction, val)
    # Safe divide
    sdiv = lambda num, div: 0 if num == 0 else num / div
    sensitivity = sdiv(tp, tp + fn)
    specificity = sdiv(tn, tn + fp)
    balanced_acc = (sensitivity + specificity) / 2
    ppv = sdiv(tp, tp + fp)
    npv = sdiv(tn, tn + fn)
    acc = sdiv(tp + tn, fp + fn + tp + tn)
    return acc, sensitivity, specificity, balanced_acc, ppv, npv


def distribution(arg: list) -> Tuple[float, float]:
    """Function to calculate mean and standard deviation from list of numbers"""
    mean = sum(arg) / len(arg)
    add = 0
    for val in arg:
        std = val - mean
        std = std * std
        add += std
    standard_deviation = math.sqrt(add / len(arg))
    return mean, standard_deviation


def cross_val(n: int, k: int = 10) -> list:
    size_set = n / k
    indexes = []
    for i in range(k):
        set_nums = list(range(math.floor(i * size_set), math.floor((i + 1) * size_set)))
        indexes.append(set_nums)
    return indexes


def reshape(arr: List[List[float]]) -> List[float]:
    arr_fix = []
    for l in arr:
        arr_fix.extend(l)
    return arr_fix


# Create machine learning model instance
def run_model(features_train, y_train, features_test, y_test) -> Tuple[float, float, float, float, float, float]:
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
    return get_predictive_values(predictions, y_test.tolist())


# Stop execution if not main
if __name__ == "__main__":
    # A list of the columns to use when running experiments
    all_columns = ['XRCC6', 'XRCC5', 'XRCC7', 'LIG4', 'LIG3', 'LIG1', 'XRCC4', 'NHEJ1', 'XRCC1', 'DCLRE1C', 'TP53BP1',
                   'BRCA1', 'BRCA2', 'EXO1', 'EXD2', 'POLM', 'POLL', 'POLQ', 'RAD50', 'MRE11', 'NBN', 'TDP1', 'RBBP8',
                   'CTBP1', 'APLF', 'PARP1', 'PARP3', 'PNKP', 'APTX', 'WRN', 'PAXX', 'RIF1', 'RAD52', 'RAD51', 'ATM', 'ATR',
                   'TP53', 'H2AX', 'ERCC1', 'ERCC4', 'RPA1', 'MSH2', 'MSH3', 'RAD1', 'MSH6', 'PMS1', 'MLH1', 'MLH3']
    columns = input("Columns separated by column (enter for all genes):\n").replace(" ", "").split(",")
    if columns == ['']:
        columns = all_columns

    start_t = time.time()
    # Import Train Data
    data_csv = pd.read_csv(csv_name, sep=",")
    data_csv.head()
    # Remove old row numbers
    data_csv = data_csv.drop("Unnamed: 0", axis=1)

    # Use 374 in ovarian+breast cancer experiment
    folds = cross_val(len(data_csv))

    # Run the experiment multiple times
    for i, fold in enumerate(folds):
        # 10-fold cross validation
        train = data_csv.drop(fold, axis=0)
        test = data_csv.loc[fold]

        train_features = train[columns]
        train_y = train[y_col]

        test_features = test[columns]
        test_y = test[y_col]
        for rep in range(repetitions):
            acc, sens, spec, bal_acc, ppv, npv = run_model(features_train=train_features, y_train=train_y,
                                                           features_test=test_features, y_test=test_y)
            print('Values (TESTING):', acc, bal_acc)

            # Adds accuracy to array
            model_acc[i].append(acc)
            model_sens[i].append(sens)
            model_spec[i].append(spec)
            model_bal[i].append(bal_acc)
            model_ppv[i].append(ppv)
            model_npv[i].append(npv)

    for i in range(10):
        mean_acc, std_acc = distribution(model_acc[i])
        mean_bal, std_bal = distribution(model_bal[i])
        mean_sens, std_sens = distribution(model_sens[i])
        mean_spec, std_spec = distribution(model_spec[i])
        mean_ppv, std_ppv = distribution(model_ppv[i])
        mean_npv, std_npv = distribution(model_npv[i])
        print("Fold Index:", i)
        print(mean_acc, mean_sens, mean_spec, mean_bal, mean_ppv, mean_npv)
        print(std_acc, std_sens, std_spec, std_bal, std_ppv, std_npv)
    print("Repetitions: " + str(repetitions))

    mean_acc, std_acc = distribution(reshape(model_acc))
    mean_bal, std_bal = distribution(reshape(model_bal))
    mean_sens, std_sens = distribution(reshape(model_sens))
    mean_spec, std_spec = distribution(reshape(model_spec))
    mean_ppv, std_ppv = distribution(reshape(model_ppv))
    mean_npv, std_npv = distribution(reshape(model_npv))
    print("Final Averages")
    print("Acc", "Sens", "Spec", "BalAcc", "PPV", "NPV")
    print(mean_acc, mean_sens, mean_spec, mean_bal, mean_ppv, mean_npv)
    print(std_acc, std_sens, std_spec, std_bal, std_ppv, std_npv)
    print(f"dur_t = {time.time() - start_t}")

