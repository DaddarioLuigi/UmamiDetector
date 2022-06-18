"""

Predizione del glutammato con X = [{AmminoAcids}+{Additives}] and y = [glutamate]

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

#open csv
df = pd.read_csv("C:/Users/info/PycharmProjects/UmamiDetector/data/umami_dataset_update.csv", delimiter=';')

#isolo il target
y = df['glutamate']

#cancello le colonne che non mi servono per la predizione
columns_to_be_deleted = ['name', 'region','group','source','glutamate']
df.drop(columns_to_be_deleted, axis=1, inplace=True)

X = df

#split del training set
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Utilizziamo questo regressore perchè ci consente di operare sui dati anche se vi sono dati mancanti
model = HistGradientBoostingRegressor().fit(X, y)
prediction = model.predict(X)

mae_train = mean_absolute_error(y, prediction)

print(f'MAE TRAIN: {mae_train}')

#non ho abbastanza dati per capire se si sta comportando bene.
X_test = np.matrix([0,0,0,26.48,32.65,71.53,180.52,107.84,75.36,4.8,19,54.54,0,2.04,36.5,31.18,17.92,36.56,20.6,55.42,105.28,23,2])
#prediction_test = np.matrix([])
prediction_test = model.predict(X_test)

print(f'{prediction_test}')




