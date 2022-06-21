# UmamiDetector


Lo scopo del progetto è quello di studiare la formazione del gusto nel cibo. Per fare questo mi avvalgo di un dataset trovato su kaggle, denominato “Umami (savoriness) in food”. Assumiamo che il gusto che vogliamo studiare sia “umami”, e, attenendoci a quanto letto sulle info del dataset e grazie ad altre rilevanze scientifiche sappiamo che:
“Amino acids in foods have two types. The first type is amino acids that are joined to build proteins. The other type is free amino acids which are dispersed. While protein has no taste, free amino acids have a taste. Free glutamate is one of the representative umami taste substances. “
-Gonzalo Recio in kaggle


1)	$U(x) = fa + (IsG)$  Umami taste
Aldilà dell'utilizzo corretto degli operatori dell'equazione definita sopra, parto con l'ipotesi che il gusto umami possa essere formato dall'interazione di Inosinato e Guanilato (s rappresenta l'operatore che definisce l'interazione), dagli amminoacidi liberi e da una f, che potrebbe essere una costante o un'altra funzione da scoprire.

Il progetto è stato realizzato utilizzando Python e scikit-learn, libreria per machine e deep learning.
Il dataset di kaggle ha diversi problemi, primo tra tutti la mancanza di molti valori, soprattutto quelli relativi 
ai due additivi che, presumibilmente sono direttamente incidenti sul valore del glutammato, che abbiamo detto essere rappresentativo del gusto Umami, cioè quello che stiamo cercando.

Ho organizzato il progetto principalmente in 3 cartelle principali:

**1) Data**
**2) Models**
**3) Visualization**

**Data**: Nella prima cartella ho collocato tutti i files che mi hanno permesso di modificare il dataset per poter utilizzare i modelli
di classificazione e regressione. Per tutti i valori mancanti ho utilizzato un Imputer KNN e poi ho 
segmentato i dati per poter raggruppare le tipologie di cibo. In modo del tutto astratto ho immaginato 
che ogni cluster potesse rappresentare una certa qualità di cibo. (cluster 1: pochissimo umami --> cluster 5: tanto umami)

**Models**: Nella cartella models ho utilizzato "**FoodClassificator.py**" per i task di classificazione del cibo. Ho utilizzato 
come target i cluster creati dal modello precedente e come classificatore ho utilizzato un DecisionTree. Per quanto 
riguarda il file "**glutamatePrediction.py**", in questo file ho utilizzato diversi regressori, che poi ho confrontato, nel powerpoint
di presentazione spiego tutto nel dettaglio. "**HistGradientClassification.py**" e "**HistGradientRegression.py**" sono 
rispettivamente due files in cui ho fatto classificazione regressione con modelli compositi, per poter lavorare
con i dati raw, senza la necessità di applicare imputers.

**Visualization**: Nella cartella visualization vi sono semplicemente tutti i plot realizzati per l'analisi dei 
dati rappresentati dal dataset.

**Possibili evoluzioni**: L'evoluzione principale per quanto riguarda questo dataset è sicuramente relativa al miglioramento dello stesso.
I risultati, soprattutto della predizione non sono molto confortanti, anche se forniscono dei sintomi di giustificazione della tesi.
Un dataset completo e senza troppi valori mancanti potrebbe aiutare sicuramente molto, inoltre, per ora, mi limito a dire che si potrebbe 
pensare di estendere la questione a tutti gli altri gusti e magari, costruito un modello completo, pensare a sistemi basati su conoscenza,
capaci di ragionare vincolati ai risultati appresi durante il lavoro di machine learning.


