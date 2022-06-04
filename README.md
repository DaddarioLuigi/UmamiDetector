# UmamiDetector


Lo scopo del progetto è quello di studiare la formazione del gusto nel cibo. Per fare questo mi avvalgo di un dataset trovato su kaggle, denominato “Umami (savoriness) in food”. Assumiamo che il gusto che vogliamo studiare sia “umami”, e, attenendoci a quanto letto sulle info del dataset noi sappiamo che:
“Amino acids in foods have two types. The first type is amino acids that are joined to build proteins. The other type is free amino acids which are dispersed. While protein has no taste, free amino acids have a taste. Free glutamate is one of the representative umami taste substances. “
-Gonzalo Recio in kaggle
1)	$P(x) = fa + (IsG)$  Umami taste
Allora, innanzitutto non sono sicuro che sia possibile rappresentare questa cosa con l’operatore +, tuttavia mi concentro sulla mia tesi per poter raggiungere un risultato tangibile.
Sul dataset ci sono svariati problemi, quello che dobbiamo fare è innanzitutto elencare i problemi dello stesso, per poter capire il lavoro da svolgere.
1)	Valori mancanti: Posso ipotizzare l’utilizzo di una rete bayesiana, che comunque, a prescindere dal numero dei valori, riesce in ogni caso a calcolare la regressione. Una possibile alternativa potrebbe essere quella di utilizzare gli imputers, che possono sembrare stupidi ma in questo caso ci possono aiutare a confrontare i vari risultati.
2)	Mancanza di una classificazione. Numericamente sappiamo quanto vale il glutammato, che assumeremo essere l’indicatore numerico dell’umami, ma non sappiamo classificare i cibi sulla base di quanto valga lo stesso. Dobbiamo clusterizzare.
Possiamo ipotizzare di aggiungere altri problemi.
Cosa fare?
1)	Script python per poter assegnare se un cibo è buono o no, sulla base della quantità di glutammato.
Scala: cattivo, mediocre, accettabile, buono, buonissimo, superlativo
<br/>

$<10$

<br/>

$>10 and <300$

<br />

$>300 and <500$  

<br />

$>500 and <1000$

<br />

$>1000 and <2000$

<br />

$>2000$

Dopo aver utilizzato il SimpleImputer() per colmare i vuoti all’interno del file, creo “a mano” un nuovo file csv con i valori fatti.
2)	(Sto proseguendo per la seconda strada) Dopo aver realizzato gli imputers, adesso devo effettuare la classificazione e poi effettuare 2 predizioni
1)	Sui valori di IsG -> Glutammato
2)	All values -> glutammato


