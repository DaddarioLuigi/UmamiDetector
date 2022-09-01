"""
In questo file applico le modifiche necessarie al dataset. Vengono fatte delle chiamate
a delle funzioni definite nel file dataprocess.py

"""

import pandas as pd
import dataprocess
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

df = pd.read_csv("umami_dataset.csv", delimiter=',')

#Gestione dei valori mancanti
dataFrame = dataprocess.ImputeDataframe(df)

wcss = []
for number_of_clusters in range(1,11):
    kmeans = cluster.KMeans(n_clusters=number_of_clusters, random_state = 42)
    kmeans.fit(dataFrame)
    wcss.append(kmeans.inertia_)

print(wcss)

ks = [1, 2, 3, 4, 5 , 6 , 7 , 8, 9, 10]
plt.plot(ks, wcss);
plt.xlabel("K")
plt.ylabel("wcss")
plt.axvline(3, linestyle='--', color='r')
plt.show()


#clustering (K-means)
processed_dataframe = dataprocess.MakeClusters(dataFrame)

#export del dataframe in un file csv
processed_dataframe.to_csv("_processed_dataframe.csv", index=False)



processed_dataframe[[3, 'clusters']].to_csv("relationshipgc.csv", index=False)