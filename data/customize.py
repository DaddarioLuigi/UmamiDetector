import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

#questa funzione utilizza gli imputers di sklearn per poter colmare i valori mancanti del dataset
def ImputeDataframe(DataFrame):
    transformers = [
        ['test_imputer', SimpleImputer(strategy="most_frequent"), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
    ]
    ct = ColumnTransformer(transformers)
    df = ct.fit_transform(DataFrame)
    fdf = pd.DataFrame(df) #final dataframe construction
    fdf.to_csv("imputed_values.csv", index=False) #save the result into a csv file. Remind that in the csv you can't see column name, so take a deep look

