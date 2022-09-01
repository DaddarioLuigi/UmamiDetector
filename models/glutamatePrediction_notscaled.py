"""
Questo file contiene la predizione del valore del glutammato utilizzando la regressione lineare per
poterne attribuire il valore numerico. In questo caso ovviamente stiamo utilizzando il training set
che ho modificato in modo tale da poter colmare i dati di training mancanti.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.neighbors  import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt


def mae_report(y, prediction):
    error = 0
    for n in range(10):
        error = error+mean_absolute_error(y, prediction)

    return error/10



# carico il csv
df = pd.read_csv("C:/Users/Luigi Daddario/Desktop/UmamiDetector-main/data/_processed_dataframe.csv", delimiter=',')



# target
y = df['3']

# cancello le colonne che non mi servono per la predizione'clusters'
columns_to_be_deleted = ['3', 'clusters']
df.drop(columns_to_be_deleted, axis=1, inplace=True)


# features
X = df

print(X)

# split del training set per renderlo utile all'addestratore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# REGRESSORE KNN
model = KNeighborsRegressor(n_neighbors=2)
model.fit(X_train, y_train)  # addestramento


# REGRESSIONE LINEARE
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# MLRegressor
neural_model = MLPRegressor(hidden_layer_sizes=[100], max_iter=100, tol=-1, verbose=2)
neural_model.fit(X_train, y_train)

# effettuo la predizione
p_train = model.predict(X_train)
p_test = model.predict(X_test)
linear_p_train = linear_model.predict(X_train)
linear_p_test = linear_model.predict(X_test)
neural_model_train = neural_model.predict(X_train)
neural_model_test = neural_model.predict(X_test)


knn_report = mae_report(y_train, p_train)
knn_report_test = mae_report(y_test, p_test)

linear_report = mae_report(y_train, linear_p_train)
linear_report_test = mae_report(y_test, linear_p_test)

neural_report = mae_report(y_train, neural_model_train)
neural_report_test = mae_report(y_test, neural_model_test)

print(f'KNN train (10 run): {knn_report} KNN test (10 run): {knn_report_test}')
print(f'Linear train (10 run): {linear_report} Linear test (10 run): {linear_report_test}')
print(f'Neural networks train (10 run): {neural_report} Neural networks test (10 run): {neural_report_test}')


# calcolo i residui
res_knn_train = y_train - p_train
res_knn_test = y_test - p_test
sns.scatterplot(x=y_train, y=res_knn_train)
plt.show()

res_linear_regression = y_train - linear_p_train
res_linear_regression_test = y_test - linear_p_test
sns.scatterplot(x=y_train, y=res_linear_regression)
plt.show()

res_neural_network = y_train - neural_model_train
res_neural_network_test = y_test - neural_model_test
sns.scatterplot(x=y_train, y=res_neural_network)
plt.show()

X_test_user = np.matrix(
    [0.0,0.0,77.0,730.0,510.0,820.0,530.0,0.0,730.0,500.0,1480.0,840.0,0.0,350.0,680.0,1430.0,670.0,880.0,210.0,580.0,290.0,890.0])
#prediction_test = np.matrix([])
prediction_test = model.predict(X_test_user)

print(f'{prediction_test}')
