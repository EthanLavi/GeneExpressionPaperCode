# Import Statements
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import seaborn as sns

# Clean data a bit
data_scaled = pd.read_csv("SurvivalNormal.csv", sep=",")
data_scaled = data_scaled.drop("Unnamed: 0", axis=1)
data_scaled.head()

# Dendogram
plt.figure(figsize=(100, 7))
plt.title("Dendrogram: Gene Expression & Survival")
matrix = shc.linkage(data_scaled.transpose(), method='ward', metric='euclidean')
dend = shc.dendrogram(matrix, labels=data_scaled.columns)
plt.show()

# Clustermap
rowcolors = data_scaled["SURVIVAL"].map({0: "blue", 1: "red"})
cmap = sns.clustermap(data_scaled, method='ward', metric='euclidean', row_colors=rowcolors, xticklabels=True,
                      dendrogram_ratio=0.05)
# Hide dendograms
cmap.ax_col_dendrogram.set_visible(False)
cmap.ax_row_dendrogram.set_visible(False)
cmap.ax_cbar.set_visible(False)
# Set text sizes on labels
for label in cmap.ax_row_colors.get_xticklabels():
    label.set(fontsize='x-small')
for label in cmap.ax_heatmap.get_xticklabels():
    label.set(fontsize='x-small')
for label in cmap.ax_heatmap.get_yticklabels():
    label.set(fontsize='xx-small')
cmap.fig.suptitle('Heatmap: Gene Expression & Survival')
plt.show()

