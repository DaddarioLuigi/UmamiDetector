import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import sklearn.cluster as cluster
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    kmeans = cluster.KMeans(n_clusters=5, init="k-means++")
    """
    Sto utilizzando solo queste 3 variabili per clusterizzare perchè ho assunto che il glutammato sia indicatore diretto
    di umami e che l'inosinato e il guanilato siano responsabili diretti della crescita del glutammato. Quindi, ipoteticamente:

    D(I) x D(G) x D(Gl) --> Maggiore umami

    """
    kmeans = kmeans.fit(DataFrame[[0, 1, 3]])

    # aggiungo la colonna al dataset
    DataFrame['clusters'] = kmeans.labels_

    #generating plots
    plot = sns.scatterplot(x=DataFrame[0], y=DataFrame[1], hue=DataFrame['clusters'], data=DataFrame)
    plot.set_xlabel("Inosinato")
    plot.set_ylabel("Guanilato")
    fig = plot.get_figure()
    fig.savefig("C:/Users/info/PycharmProjects/UmamiDetector/visualization/scatteradditives.png")
    plt.show()

    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111, projection='3d')

    x = DataFrame[0]
    y = DataFrame[1]
    z = DataFrame[3]

    ax.set_xlabel("Inosinato")
    ax.set_ylabel("Guanilato")
    ax.set_zlabel("Glutammato")

    ax.scatter(x, y, z, c=DataFrame['clusters'])
    fig_2.savefig("C:/Users/info/PycharmProjects/UmamiDetector/visualization/3dglutammateclustering.png")

    plt.show()

    return DataFrame

