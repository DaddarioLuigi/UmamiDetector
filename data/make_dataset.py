#In questo documento sto applicando le modifiche necessarie al dataset. Lo carico e utilizzo
#uno script per classificare iterativamente i vari cibi.

import pandas as pd
import customize
import numpy as np

df = pd.read_csv("umami_dataset.csv", delimiter=',')

#ciclo in tutte le righe del dataframe
for i in range(len(df)):
    if(df.loc[i, 'glutamate'] < 10):
        df.loc[i, 'judgment'] = "bad"
    if (df.loc[i, 'glutamate'] > 10) and (df.loc[i, 'glutamate'] < 300):
        df.loc[i, 'judgment'] = "unexceptional"
    if (df.loc[i, 'glutamate'] > 300) and (df.loc[i, 'glutamate'] < 500):
        df.loc[i, 'judgment'] = "acceptable"
    if (df.loc[i, 'glutamate'] > 500)  and (df.loc[i, 'glutamate'] < 1000):
        df.loc[i, 'judgment'] = "good"
    if (df.loc[i, 'glutamate'] > 1000) and (df.loc[i, 'glutamate'] < 2000):
        df.loc[i, 'judgment'] = "very good"
    if (df.loc[i, 'glutamate'] > 2000):
        df.loc[i, 'judgment'] = "amazing"

#print(df['judgment']) #stampo a video per visualizzare i cambiamenti

customize.ImputeDataframe(df)

df.to_csv("umami_dataset_scripted.csv") #salvo il dataset dopo aver effettuato lo script

