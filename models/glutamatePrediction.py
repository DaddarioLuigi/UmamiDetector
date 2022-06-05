"""
Questo file contiene la predizione del valore del glutammato utilizzando la regressione lineare per
poterne attribuire il valore numerico. In questo caso ovviamente stiamo utilizzando il training set
che ho modificato in modo tale da poter colmare i dati di training mancanti.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors  import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

#carico il csv
df = pd.read_csv("C:/Users/info/PycharmProjects/UmamiDetector/data/processed_dataframe.csv", delimiter=',')

#target
y = df['3']

#cancello le colonne che non mi servono per la predizione
columns_to_be_deleted = ['3','clusters']
df.drop(columns_to_be_deleted, axis=1, inplace=True)

#features
X = df

#split del training set per renderlo utile all'addestratore
X_train, X_test, y_train, y_test = train_test_split(X, y)

#REGRESSORE KNN
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train) #addestramento

#REGRESSIONE LINEARE
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


#effettuo la predizione
p_train = model.predict(X_train)
p_test = model.predict(X_test)
linear_p_train = linear_model.predict(X_train)
linear_p_test = linear_model.predict(X_test)

#calcolo mean absolute error dei dati di training e dei dati di test
mae_train = mean_absolute_error(y_train, p_train)
mae_test = mean_absolute_error(y_test, p_test)

mae_linear_train = mean_absolute_error(y_train, linear_p_train)
mae_linear_test = mean_absolute_error(y_test, linear_p_test)

print(f'Train {mae_train} Test {mae_test}')

plt.rcdefaults()
fig, ax = plt.subplots()

dati = ("mae_train", "mae_test", "mae_linear_train", "mae_linear_test")
y_pos = np.arange(len(dati))
performance = (mae_train, mae_test, mae_linear_train, mae_linear_test)
error = (mae_train, mae_test, mae_linear_train, mae_linear_test)


ax.barh(y_pos, performance, xerr=error, align='center')
ax.set_yticks(y_pos, labels=dati)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('Errori di train e test a confronto')

plt.show()


#test della predizione
#prediction_test = np.matrix([0,77,55,52,58,204,0,226,412,180,64,0,43,54,86,37,49,0,66,29,529,529])
#prediction = model.predict(prediction_test)

#print(f'{prediction}')
