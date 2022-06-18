"""
Classificazione del cibo sulla base della quantità di Umami contenuta

"""

import numpy as np
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#open csv
df = pd.read_csv("C:/Users/info/PycharmProjects/UmamiDetector/data/processed_dataframe.csv", delimiter=',')

#isolo il target
y = df['clusters']

#cancello le colonne che non mi servono per la predizione
columns_to_be_deleted = ['clusters']
df.drop(columns_to_be_deleted, axis=1, inplace=True)

#features
X = df

#split dei dati di train e dei dati di test
X_train, X_test, y_train, y_test = train_test_split(X, y)

#utilizzo un decision tree come classificatore
model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)

#predizioni
p_train = model.predict(X_train)
p_test = model.predict(X_test)

#report della classificazione
report = classification_report(y_train, p_train)
print(report)

#mostro una matrice di confusione
skplt.metrics.plot_confusion_matrix(y_train, p_train)
plt.show()

#report sui dati di test
report_test = classification_report(y_test, p_test)
print(report_test)

skplt.metrics.plot_confusion_matrix(y_test, p_test)
plt.show()


prediction_test = np.matrix([0.0,0.0,77.0,1440.0,1440.0,780.0,720.0,485.0,0.0,430.0,500.0,1060.0,970.0,40.0,470.0,830.0,1220.0,440.0,650.0,220.0,1500.0,820.0,880.0])
#prediction_test = np.matrix([])
prediction = model.predict(prediction_test)

print(f'{prediction}')