import matplotlib.pyplot as plt

# Load previous entries
dot_dat = input("Data File:\n")
f = open(dot_dat, "r")
f.readline()
lines = f.read().split("\n")
f.close()

# Make arrays
acc_plotter = []
avg_acc_plotter = []
xs = []
ys = []

# Iterate through lines
pair_best = ["", 0, 0]
for line in lines:
    # Process the line
    tokens = line.split(",", 2)
    if len(tokens) < 3:
        continue
    col = tokens[0]
    acc = float(tokens[1])
    lister = tokens[2].replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(",")

    # add the data to xs and ys
    xs.append(len(lister))
    ys.append(acc)

    # If we have a new group
    if len(lister) > pair_best[2]:
        # Reset pair_best and add the old accuracy
        pair_best[2] = len(lister)
        acc_plotter.append(pair_best[1])
        pair_best[1] = 0

    # If accuracy has improved
    if acc > pair_best[1]:
        # Include in the pair best
        pair_best[1] = acc
        pair_best[0] = col

# Add the last best accuracy and drop the invalid accuracy at the beginning
acc_plotter.append(pair_best[1])
acc_plotter.pop(0)

# Calculate group accuracies
summer, counter, group = 0, 0, 1
for i, val in enumerate(ys):
    if xs[i] == group:
        summer += val
        counter += 1
    else:
        avg_acc_plotter.append(summer / counter)
        summer = val
        counter = 1
        group += 1
avg_acc_plotter.append(summer / counter)

# Graph
xval = range(1, len(acc_plotter) + 1)
plt.xticks(xval)
plt.xlabel("n-Added Feature", fontsize=12)
plt.ylabel("Balanced Accuracy", fontsize=12)
plt.title(f"Maximizing Balanced Accuracy of Neural Network using Genes: {input('Title: ')}", fontsize=14)
plt.plot(xs, ys, 'o', alpha=0.3, color='b', label="Data Points")
plt.plot(xval, acc_plotter, color="r", label="Best Accuracies")
plt.plot(xval, avg_acc_plotter, color="b", label="Average Accuracies")
plt.legend(loc='upper right')
plt.show()

