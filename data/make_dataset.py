"""
In questo file applico le modifiche necessarie al dataset. Vengono fatte delle chiamate
a delle funzioni definite nel file dataprocess.py

"""

import pandas as pd
import dataprocess
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("umami_dataset.csv", delimiter=',')

#Gestione dei valori mancanti
dataFrame = dataprocess.ImputeDataframe(df)

#clustering (K-means)
processed_dataframe = dataprocess.MakeClusters(dataFrame)

#export del dataframe in un file csv
processed_dataframe.to_csv("processed_dataframe.csv", index=False)

#making plots
plot = sns.pairplot(data=processed_dataframe[[0, 1, 3]])
#fig = plot.get_figure()
plot.savefig("C:/Users/info/PycharmProjects/UmamiDetector/visualization/pairplot.png")
plt.show()

