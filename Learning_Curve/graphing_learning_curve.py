import matplotlib.pyplot as plt
import numpy as np

# First learning curve
f = open("text/learning_curve_log.csv", "r")
data = [x.strip().split(",") for x in f.readlines()]
data.pop(0)  # removing header
f.close()

increments = [float(x[1]) for x in data]
train_score = [float(x[2]) for x in data]
test_score = [float(x[3]) for x in data]

# Graph scores
plt.plot(increments, train_score, marker='o', label="Average Training Score")
plt.plot(increments, test_score, marker='o', label="Average Testing Score")
plt.xticks(increments)
plt.yticks(np.arange(0.3, 1, 0.1))
plt.ylabel("Balanced Accuracy")
plt.xlabel("Training Examples")
plt.title("Learning Curve Logistic Regression Model")
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Second learning curve
f = open("output.csv", "r")
data = [x.strip().split(",") for x in f.readlines()]
data.pop(0)  # removing header
f.close()

increments = [float(x[0]) for x in data]
train_score = [float(x[1]) for x in data]
test_score = [float(x[2]) for x in data]

# Graph scores
plt.plot(increments, train_score, marker='o', label="Average Training Score")
plt.plot(increments, test_score, marker='o', label="Average Testing Score")
plt.xticks(increments)
plt.yticks(np.arange(0.3, 1, 0.1))
plt.ylabel("Balanced Accuracy")
plt.xlabel("Training Examples")
plt.title("Learning Curve Neural Network Model: Top 3 Feature Selection")
plt.legend(loc='lower right')
plt.grid()
plt.show()


# Third learning curve
f = open("output_all.csv", "r")
data = [x.strip().split(",") for x in f.readlines()]
data.pop(0)  # removing header
f.close()

increments = [float(x[0]) for x in data]
train_score = [float(x[1]) for x in data]
test_score = [float(x[2]) for x in data]

# Graph scores
plt.plot(increments, train_score, marker='o', label="Average Training Score")
plt.plot(increments, test_score, marker='o', label="Average Testing Score")
plt.xticks(increments)
plt.yticks(np.arange(0.3, 1, 0.1))
plt.ylabel("Balanced Accuracy")
plt.xlabel("Training Examples")
plt.title("Learning Curve Neural Network Model: All")
plt.legend(loc='lower right')
plt.grid()
plt.show()


# Fourth learning curve
f = open("output_5.csv", "r")
data = [x.strip().split(",") for x in f.readlines()]
data.pop(0)  # removing header
f.close()

increments = [float(x[0]) for x in data]
train_score = [float(x[1]) for x in data]
test_score = [float(x[2]) for x in data]

# Graph scores
plt.plot(increments, train_score, marker='o', label="Average Training Score")
plt.plot(increments, test_score, marker='o', label="Average Testing Score")
plt.xticks(increments)
plt.yticks(np.arange(0.3, 1, 0.1))
plt.ylabel("Balanced Accuracy")
plt.xlabel("Training Examples")
plt.title("Learning Curve Neural Network Model: Top 5 Feature Selection")
plt.legend(loc='lower right')
plt.grid()
plt.show()

