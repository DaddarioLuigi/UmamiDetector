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
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.neighbors  import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt

#carico il csv
df = pd.read_csv("C:/Users/info/PycharmProjects/UmamiDetector/data/processed_dataframe.csv", delimiter=',')

#target
y = df['3']

#cancello le colonne che non mi servono per la predizione'clusters'
columns_to_be_deleted = ['3','clusters']
df.drop(columns_to_be_deleted, axis=1, inplace=True)

#features
X = df

#split del training set per renderlo utile all'addestratore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#REGRESSORE KNN
model = KNeighborsRegressor(n_neighbors=10)
model.fit(X_train, y_train) #addestramento

#REGRESSIONE LINEARE
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#BAYESIAN RIDGE
bayesian_model = BayesianRidge(compute_score=True, n_iter=30)
bayesian_model.fit(X_train, y_train)

#MLRegressor
neural_model = MLPRegressor(hidden_layer_sizes=[100], max_iter=10000, tol=-1, verbose=2)
neural_model.fit(X_train, y_train)

#effettuo la predizione
p_train = model.predict(X_train)
p_test = model.predict(X_test)
linear_p_train = linear_model.predict(X_train)
linear_p_test = linear_model.predict(X_test)
bayesian_p_train = bayesian_model.predict(X_train)
bayesian_p_test = bayesian_model.predict(X_test)
neural_model_train = neural_model.predict(X_train)
neural_model_test = neural_model.predict(X_test)

#calcolo mean absolute error dei dati di training e dei dati di test
mae_train_knn = mean_absolute_error(y_train, p_train)
mae_test_knn = mean_absolute_error(y_test, p_test)

mae_linear_train = mean_absolute_error(y_train, linear_p_train)
mae_linear_test = mean_absolute_error(y_test, linear_p_test)

mae_bayesian_train = mean_absolute_error(y_train, bayesian_p_train)
mae_bayesian_test = mean_absolute_error(y_test, bayesian_p_test)

mae_neuralmodel_train = mean_absolute_error(y_train, neural_model_train)
mae_neuralmodel_test = mean_absolute_error(y_test, neural_model_test)


print(f'Train KNN {mae_train_knn} Test KNN {mae_test_knn}')
print(f'Train Linear Regression {mae_linear_train} Test Linear Regression {mae_linear_test}')
print(f'Train Bayesian {mae_bayesian_train} Test Bayesian {mae_bayesian_test}')
print(f'Train Neural Network {mae_neuralmodel_train} Test Neural Network {mae_neuralmodel_test}')

#calcolo i residui
res_knn_train = y_train - p_train
res_knn_test = y_test - p_test
sns.scatterplot(x=y_train,y=res_knn_train)
plt.show()

res_linear_regression = y_train - linear_p_train
res_linear_regression_test = y_test - linear_p_test
sns.scatterplot(x=y_train,y=res_linear_regression)
plt.show()

res_bayesian = y_train - bayesian_p_train
res_bayesian_test = y_test - bayesian_p_test
sns.scatterplot(x=y_train,y=res_bayesian)
plt.show()

res_neural_network = y_train - neural_model_train
res_neural_network_test = y_test - neural_model_test
sns.scatterplot(x=y_train,y=res_neural_network)
plt.show()


X_test_user = np.matrix([0.0,0.0,77.0,1440.0,780.0,720.0,485.0,0.0,430.0,500.0,1060.0,970.0,40.0,470.0,830.0,1220.0,440.0,650.0,220.0,1500.0,820.0,880.0])
#prediction_test = np.matrix([])
prediction_test = neural_model.predict(X_test_user)

print(f'{prediction_test}')