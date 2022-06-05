"""
Classificazione del cibo sulla base della quantità di Umami contenuta
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#open csv
df = pd.read_csv("C:/Users/info/PycharmProjects/UmamiDetector/data/processed_dataframe.csv", delimiter=',')

#isolo il target
y = df['clusters']

#cancello le colonne che non mi servono per la predizione
columns_to_be_deleted = ['clusters']
df.drop(columns_to_be_deleted, axis=1, inplace=True)

#features
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

#calcolo dell'accuratezza (modello piu semplice)
acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print(f'Train {acc_train} Test {acc_test}')

#test della predizione
prediction_test = np.matrix([0, 77, 579.2, 4810.6, 308, 229.6, 0, 0, 0, 179.2, 660.8, 638.4, 0, 408.8, 408.8, 968.8, 0, 0, 0, 812, 280, 0,0])
#prediction_test = np.matrix([])
prediction = model.predict(prediction_test)

print(f'{prediction}')