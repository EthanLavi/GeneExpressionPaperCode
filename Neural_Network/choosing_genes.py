# Import
import os.path
import pandas as pd
import neural_model as neural

# Modify env
neural.repetitions = 3

# Import Train Data
data_csv = pd.read_csv(neural.csv_name, sep=",")
data_csv.head()
# Remove old row numbers
data_csv = data_csv.drop("Unnamed: 0", axis=1)
folds = neural.cross_val(len(data_csv))
columns = ['XRCC6', 'XRCC5', 'XRCC7', 'LIG4', 'LIG3', 'LIG1', 'XRCC4', 'NHEJ1', 'XRCC1', 'DCLRE1C', 'TP53BP1',
           'BRCA1', 'BRCA2', 'EXO1', 'EXD2', 'POLM', 'POLL', 'POLQ', 'RAD50', 'MRE11', 'NBN', 'TDP1', 'RBBP8', 'CTBP1',
           'APLF', 'PARP1', 'PARP3', 'PNKP', 'APTX', 'WRN', 'PAXX', 'RIF1', 'RAD52', 'RAD51', 'ATM', 'ATR', 'TP53',
           'H2AX', 'ERCC1', 'ERCC4', 'RPA1', 'MSH2', 'MSH3', 'RAD1', 'MSH6', 'PMS1', 'MLH1', 'MLH3']
best_col = []
acc_plotter = []

# Make a file for data entry
datafile = input("Choose file for data entry:\n")
if os.path.exists(datafile):
    # Load previous entries
    f = open(datafile, "r")
    f.readline()
    lines = f.read().split("\n")
    f.close()

    pair_best = ["", 0, 0]
    for line in lines:
        tokens = line.split(",", 2)
        if len(tokens) < 3:
            continue
        col = tokens[0]
        acc = float(tokens[1])
        lister = tokens[2].replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(",")

        if len(lister) > pair_best[2]:
            pair_best[2] = len(lister)
            best_col.append(pair_best[0])
            acc_plotter.append(pair_best[1])
            pair_best[1] = 0

        if acc > pair_best[1]:
            pair_best[1] = acc
            pair_best[0] = col

    best_col.append(pair_best[0])
    acc_plotter.append(pair_best[1])

    best_col.pop(0)
    acc_plotter.pop(0)
else:
    # Create the file
    f = open(datafile, "w")
    f.write("Column,Acc,Experiment_Columns\n")
    f.close()

while len(best_col) != 35:
    best_choice = None
    best_acc = 0

    # Do experiment with one-add columns
    for col in columns:
        if col in best_col:
            continue
        current_col = best_col.copy()
        current_col.append(col)
        col_balacc = []

        for i, fold in enumerate(folds):
            # 10-fold cross validation
            train = data_csv.drop(fold, axis=0)
            test = data_csv.loc[fold]

            train_features = train[current_col]
            train_y = train[neural.y_col]

            test_features = test[current_col]
            test_y = test[neural.y_col]
            for rep in range(neural.repetitions):
                # Run the experiment multiple times
                acc, sens, spec, bal_acc, ppv, npv = neural.run_model(features_train=train_features, y_train=train_y,
                                                                      features_test=test_features, y_test=test_y)
                col_balacc.append(bal_acc)

        exp_acc = sum(col_balacc) / len(col_balacc)
        f = open(datafile, "a")
        f.write(f"{col},{exp_acc},{str(current_col)}\n")
        f.close()
        if exp_acc > best_acc:
            best_acc = exp_acc
            best_choice = col

    # Add the best column to list
    best_col.append(best_choice)
    acc_plotter.append(best_acc)

print(best_col)
print(acc_plotter)

