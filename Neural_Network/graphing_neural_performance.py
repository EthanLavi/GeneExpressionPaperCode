import matplotlib.pyplot as plt
import numpy as np
import math

bar_width = 0.2

test_names = ["All Genes", "Neural Network Feature Selection", "Logistic Regression Feature Selection"]
base = [.483381437, .57762513, .53225336]
base_pos = np.arange(len(test_names))
augmented = [.46320638, .490976663, .490041479]
aug_pos = [x + bar_width for x in base_pos]

# Max standard deviation observed was 0.1
# Calculating 95% confidence interval using this
std = 0.1
ci_95_dev = 1.96 * (std / math.sqrt(10 * 10))  # 10 * 10 because 10-folds with 10 repetitions each
plt.bar(base_pos, base, yerr=ci_95_dev, capsize=10, ecolor='black', width=bar_width, color="#1167b1", label="Only Ovarian Cancer Data")
plt.bar(aug_pos, augmented, yerr=ci_95_dev, capsize=10, ecolor='black', width=bar_width, color="#e9724d", label="Augmented With Breast Cancer Data")
for i, v in enumerate(base):
    plt.text(i, round(v, 4) + 0.002, round(v, 4), ha='center', bbox=dict(facecolor='white', edgecolor='white', pad=1.0))
for i, v in enumerate(augmented):
    plt.text(i + bar_width, round(v, 4) + 0.002, round(v, 4), ha='center', bbox=dict(facecolor='white', edgecolor='white', pad=1.0))
plt.xlabel("Gene Tests", fontsize=12)
plt.ylabel("Balanced Accuracy", fontsize=12)
plt.title("Comparing accuracy when training with augmented dataset: Survival", fontsize=14)
plt.legend()
plt.xticks((base_pos + aug_pos) / 2, test_names)
plt.ylim([.45, .65])
plt.show()

