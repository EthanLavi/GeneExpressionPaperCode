# Import Statements
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

data_scaled = pd.read_csv("SurvivalNormal.csv", sep=",")
# Remove row numbers
data_scaled = data_scaled.drop("Unnamed: 0", axis=1)
data_scaled.head()

print(data_scaled)
print(data_scaled.columns)

# Create Graphs
plt.figure(figsize=(100, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.show()

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_scaled)

fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(data_scaled.corr(), center=0, annot=False, ax=ax)
plt.show()

sns.clustermap(data_scaled, metric='correlation', xticklabels=True)
plt.show()

