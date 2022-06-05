#In questo documento sto applicando le modifiche necessarie al dataset. Lo carico e utilizzo
#uno script per classificare iterativamente i vari cibi.

import pandas as pd
import dataprocess
import numpy as np

df = pd.read_csv("umami_dataset.csv", delimiter=',')

#applico un algoritmo per gestire i valori mancanti
dataFrame = dataprocess.ImputeDataframe(df)
#clustering (KNN)
processed_dataframe = dataprocess.MakeClusters(dataFrame)
processed_dataframe.to_csv("processed_dataframe.csv", index=False)


