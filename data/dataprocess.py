import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import sklearn.cluster as cluster
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

#questa funzione utilizza gli imputers di sklearn per poter colmare i valori mancanti del dataset
from sklearn.impute import KNNImputer

#utilizzo un imputer di KNN
def ImputeDataframe(DataFrame):
    imputer = KNNImputer(n_neighbors=2)
    columns_to_be_deleted = ['name', 'region', 'group', 'source']
    DataFrame.drop(columns_to_be_deleted, axis=1, inplace=True)
    df = imputer.fit_transform(DataFrame)
    df = pd.DataFrame(df) #final dataframe construction
    return df


def MakeClusters(DataFrame):
    kmeans = cluster.KMeans(n_clusters=3, init="k-means++")

    kmeans = kmeans.fit(DataFrame)

    # aggiungo la colonna al dataset
    DataFrame['clusters'] = kmeans.labels_

    #faccio due join plot guanilato-glutammato inosinato-glutammato
    plot = sns.scatterplot(x=DataFrame[0], y=DataFrame[3], hue=DataFrame['clusters'], data=DataFrame)
    plot.set_xlabel("Inosinato")
    plot.set_ylabel("Glutammato")
    fig = plot.get_figure()
    fig.savefig("C:/Users/Luigi Daddario/Desktop/UmamiDetector-main/visualization/inosatojoinglutammato.png")
    plt.show()

    # faccio due join plot guanilato-glutammato inosinato-glutammato
    plot = sns.scatterplot(x=DataFrame[1], y=DataFrame[3], hue=DataFrame['clusters'], data=DataFrame)
    plot.set_xlabel("Guanilato")
    plot.set_ylabel("Glutammato")
    fig = plot.get_figure()
    fig.savefig("C:/Users/Luigi Daddario/Desktop/UmamiDetector-main/visualization/guanilatojoinglutammato.png")
    plt.show()

    # faccio due join plot guanilato-glutammato inosinato-glutammato
    plot = sns.scatterplot(x=DataFrame[4], y=DataFrame[3], hue=DataFrame['clusters'], data=DataFrame)
    plot.set_xlabel("Acido Aspartico")
    plot.set_ylabel("Glutammato")
    fig = plot.get_figure()
    fig.savefig("C:/Users/Luigi Daddario/Desktop/UmamiDetector-main/visualization/guanilatojoinglutammato.png")
    plt.show()

    # faccio due join plot guanilato-glutammato inosinato-glutammato
    plot = sns.scatterplot(x=DataFrame[11], y=DataFrame[3], hue=DataFrame['clusters'], data=DataFrame)
    plot.set_xlabel("Analina")
    plot.set_ylabel("Glutammato")
    fig = plot.get_figure()
    fig.savefig("C:/Users/Luigi Daddario/Desktop/UmamiDetector-main/visualization/guanilatojoinglutammato.png")
    plt.show()

    return DataFrame
