import subprocess
from typing import List, Tuple
import sys

format = ["Logistic Regression Balanced Accuracy", "Decision Tree Balanced Accuracy", 
          "NaiveBayes Balanced Accuracy", "SVM Linear Balanced Accuracy", "SVM Radial Balanced Accuracy", 
          "SVM Sigmoid Balanced Accuracy", "SVM Polynomial Balanced Accuracy", "Significant P-Values"]
class_values = ["LogRegBalAcc", "TreeBalAcc", "NaiveBalAcc", "SVMLinBalAcc", "SVMRadBalAcc", "SVMSigBalAcc", "SVMPolyBalAcc", "SigPValCount"]

class ResultFormat:
    LogRegBalAcc: float
    TreeBalAcc: float
    NaiveBalAcc: float
    SVMLinBalAcc: float
    SVMRadBalAcc: float
    SVMSigBalAcc: float
    SVMPolyBalAcc: float
    SigPValCount: int

    def __init__(self, format_arr=None) -> None:
        if format_arr is None:
            return
        for i, v in enumerate(format_arr):
            exec(f"self.{class_values[i]} = {v}")

    def print_values(self, prefix: str):
        print(prefix)
        for i, key in enumerate(format):
            val = eval(f"self.{class_values[i]}")
            print(f"{key} {val}")

def tokenize(out: str) -> ResultFormat:
    """Process the output of the Rscript to compile it"""
    # Clean the output string to make it readable
    out = out.replace("\"", "")
    out = out.replace("[1]", "")
    tokens = out.split("\\n")
    bal_accuracies = []
    p_values = []
    for line in tokens:
        line = line.strip()
        if line.__contains__("Balanced Accuracy"):
            bal_accuracies.append(line)
        elif line.__contains__("TRUE") or line.__contains__("FALSE"):
            p_values.append(line)

    # Process balanced accuracy array
    format = []
    for line in bal_accuracies:
        tokens = line.split()
        value = 0
        for token in tokens:
            try:
                if type(eval(token)) != str:
                    value = eval(token)
            except Exception:
                continue
        format.append(value)

    # Process P-values
    count = 0
    for line in p_values:
        tokens = line.split()
        datums = []
        for token in tokens:
            try:
                if type(eval(token)) != str:
                    datums.append(eval(token))
            except Exception:
                continue
        if datums[1] <= 0.05:
            count += 1
    format.append(count)
    return ResultFormat(format)


def statify(format_arrays: List[ResultFormat]) -> Tuple[ResultFormat, ResultFormat]:
    """Extract average and 95% percentile out of results"""
    res_avg = ResultFormat()
    res_95p = ResultFormat()

    for member in class_values:
        arr = []
        for form in format_arrays:
            arr.append(eval(f"form.{member}"))
        arr.sort()
        index = int(len(arr) * 0.95) - 1
        exec(f"res_avg.{member} = sum(arr) / len(arr)")
        exec(f"res_95p.{member} = arr[index]")
    return res_avg, res_95p
        
results = []
if len(sys.argv) != 2:
    print("Must provide a n-value for the number of trials (usage: python control_trials.py N)")
    exit(1)
trial_count = eval(sys.argv[1])
for i in range(trial_count):
    pipe = subprocess.Popen(["Rscript", "control_analysis.R"], stdout=subprocess.PIPE)
    output = tokenize(str(pipe.communicate()[0]))
    results.append(output)
    print(f"Trial {i+1} conducted!")

res_avg, res_95p = statify(results)
res_avg.print_values("Result Averages")
print("------------")
res_95p.print_values("Result 95% Percentile")
print(f"Results based on {trial_count} trials")