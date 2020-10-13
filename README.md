# TD - Clustering des données IRIS

## Ce qu'il vous faut 

- sklearn
- matplotlib
- seaborn
- pandas
- numpy

## Les données

Pour charger les données Iris, *sklearn* propose une fonction *sklearn.datasets.load_iris*. Cela retourne un dictionnaire que vous pouvez interroger en utilisant *.keys()* sur l'objet retourné.

Ensuite, l'idéal est de charger les données *X* dans *pandas*, comme par exemple : 
 ```
 X=load_iris().data
 pdX = pd.DataFrame(X)
 pdX.columns=load_iris().feature_names
 pdX.head()
 ```
Créer un autre dataframe pour les classes *y* (load_iris().target).
Puis, en utilisant la fonction *concat* de pandas, fusionner les deux dataframes en un seul.

## Stats et corrélations sur les données

Il est possible d'appeler *describe* sur un dataframe pandas pour obtenir des statistiques descriptives simples sur chacun des attributs/variables/features utilisés pour décrire les données.

Nous allons ensuite étudier les corrélations entre nos variables entre elles, mais également entre nos variables et la classe de nos données. Pour faire cela proprement, il s'agit de transformer notre colonne classe représentée par une variable discrète à valeurs dans {0, 1, 2} en trois colonnes booléennes indiquant pour chacune l'appartenance à la classe ou non.
```
df_bool = df.copy(deep=True)
for i in range(df.classe.min(), df.classe.max() +1):
    df_bool[data.target_names[i]]= (df.classe == i)
del df_bool['classe']
df_bool.head()
```

Ensuite on peut appeler *corr* sur le dataframe ainsi créé pour obtenir les résultats et utiliser la fonctionnalité *heatmap* de seaborn pour visualiser ces corrélations.

## Clustering et évaluation en utilisant les étiquettes connues, i.e. évaluation externe/extrinsèque

En utilisant les fonctions suivantes : 
```
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
```
Faire le clustering des données Iris en utilisant l'agorithme des k-moyennes et en faisant varier le nombre de clusters *k* entre 2 et 10. Pour chaque *k*, calculer l'adjusted rand score (version revue du Rand Index vue en cours, à maximiser). Faire un plot du rand index en fonction de *k* (scatter plot), et observer les résultats.


## Clustering et évaluation sans utiliser les étiquettes, i.e. évaluation interne/intrinséque : seulement avec la structure interne des clusters

Suivre la même procédure qu'à la question précédente, mais utiliser les critères internes de Davies-Bouldin (à minimiser) et de Calinski Harabasz (à maximiser) dans *sklearn.metrics*. Quels sont les résultats obtenus ?

## Visualisation en deux dimensions (réduction de dimensions par PCA) des clusters et de leurs centres

Faire un clustering avec les k-means en utilisant 3 clusters.
En utilisant l'ACP, qui est une technique de réduction de dimensions, réduire les données (initialement décrites par 4 attributs) pour les représenter sur 2 dimensions afin de pouvoir visualiser le résultat du clustering. 
`from sklearn.decomposition import PCA`
Pour cela, faire un scatter plot des données réduites par ACP (fit and transform) et définisser la couleur de chaque point en fonction du cluster obtenu par la méthodes des k-moyennes.

Ensuite, ajouter pour chaque cluster son centre en noir afin de les visualiser.

## Étiquetage

Afin de visualiser quelles sont les attributs les plus descriptifs, et les plus typiques des clusters, implémenter la feature precision, le feature recall et la feature f-mesure et calculer ces valeurs pour chaque cluster et chaque attribut.

Avant de procédure, et afin d'éviter un effet d'échelle, procéder à un rescaling des données : `from sklearn.preprocessing import MinMaxScaler`

Pour calculer ces valeurs, la fonction *numpy.sum* qui permet de calculer la somme des colonnes, des lignes, ou de la totalité des données d'une matrice vous sera utile. De même, il est possible de filtrer la matrice *X* des données en fonction de la classe *k* du vecteur de classes *y* en utilisant `X[y == k]`.


# TD - Word embeddings pour le résumé automatique

## Embeddings pré-appris

Pour commencer à jouer avec des embeddings, le plus simple consiste à utiliser des vecteurs déjà appris sur de grands corpus. Nous allons ainsi utiliser les vecteurs pré-entraînés de Jean-Philippe Fauconnier, je vous propose de télécharger les vecteurs de dimension *1000* appris avec *Skip-gram* sur un corpus *Wikipedia* français non lemmatisé : [télécharger les vecteurs](http://embeddings.net/frWiki_no_lem_no_postag_no_phrase_1000_skip_cut200.bin).

### Comment les utiliser ?

Si vous ne l'avez pas fait, installer gensim avec `pip install --user gensim`

Ensuite  : 
```
from gensim.models import KeyedVectors
wv_from_text = KeyedVectors.load_word2vec_format('frWiki_no_lem_no_postag_no_phrase_1000_skip_cut200.bin', binary=True)
wv_from_text.most_similar("sarthe")
[('mayenne', 0.584618091583252), ('loire', 0.561012864112854), ('mamers', 0.550373375415802), ('maine', 0.5418317914009094), ('orne', 0.494168758392334), ('oudon', 0.46164029836654663), ('fresnay', 0.44641953706741333), ('ambrières', 0.44309335947036743), ('allonnes', 0.43901848793029785), ('bazouges', 0.41962945461273193)]
```

Avec la fonction `most_similar`, vous venez d'expérimenter l'intérêt des embeddings pour explorer les similarités sémantiques. En cours, nous avons vu que c'était l'une des forces de ces approches, de rapprocher dans l'espace appris les mots qui co-occurrent, et qui donc ont un lien de sens (synonymie, hyperonymie, antonymie, etc). Mais nous avons aussi parlé analogie ! 

### Résoudre votre première analogie

Comment fonctionne l'analogie ? En français, on va par exemple dire "Paris est à la France ce que Varsovie est à la Pologne". Si l'on cherche à questionner quelqu'un pour qu'iel résolve cette analogie, nous pouvons ainsi dire "Qu'est ce qui est à la Pologne ce que Paris est à la France ?". La réponse est alors *Varsovie*. En pratique, avec nos embeddings, nous allons réaliser l'opération suivante : *paris* - *france* + *pologne* en espérant que le vecteur le plus similaire au résultat de cette opération soit celui de *varsovie*.
C'est le cas : 
```
>>> wv_from_text.most_similar_cosmul(positive=['paris', 'pologne'], negative=['france'])
[('varsovie', 0.8835111856460571), ('voïvodie', 0.8167898654937744), ('polonaise', 0.8139687776565552), ('polonais', 0.8031845688819885), ('kazimierz', 0.7687076330184937), ('polonaises', 0.7532544732093811), ('voïvodies', 0.751475989818573), ('juliusz', 0.7506932616233826), ('tadeusz', 0.740206778049469), ('janusz', 0.7386343479156494)]
```
Imaginez d'autres analogies, formulez les, et postez les sur le mattermost si le résultat est cool.


## L'approche TextRank pour le résumé automatique

Pour ce travail, vous avez besoin de : 
- scipy, numpy
- gensim
- nltk
- networkx

L'approche *Textrank* pour le résumé automatique est une baseline très efficace et très communément utilisée **[Mihalcea et Tarau]**. L'approche *Textrank* est une approche de résumé automatique dite **extractive** : il s'agit de résumer un document via **l'extraction** de phrases considérées comme caractéristiques du contenu du document. *Textrank* permet également d'extraire les mots-clés pour un document afin de l'indexer, mais nous nous concentrons ici sur l'approche de résumé automatique.
 
Cette approche est basée sur l'algorithme du *Pagerank*, l'algorithme qui a notamment rendu célèbre le moteur de recherche Google **[Brin et Page]**. Ce dernier  s'applique sur des données structurées sous forme de graphe, et la construction de ce graphe est donc l'une des premières étapes de l'approche *Textrank* que nous détaillons ci-après : 
1. Séparation du document en phrases (`nltk.sent_tokenize` vous aidera) ;
2. Nettoyage des phrases (lowercase avec `.lower()`, suppression ponctuation avec `table = str.maketrans(dict.fromkeys(string.punctuation))`)
3. Représentation vectorielle des phrases avec **word2vec** : chaque phrase est représentée par la moyenne des vecteurs des mots (`.get_vector(mot)` avec l'objet gensim) qui la composent. Certains mots de la phrase peuvent ne pas être dans le vocabulaire du modèle pré-appris (exception `KeyError`), on ne tient pas compte de ces mots pour le calcul.
4. Calcul de la similarité cosine (`scipy.spatial.distance.cosine`) de chacune des paires de phrases : matrice de similarité ;
5. Création d'un graphe valuée en utilisant la matrice de similarité (`networkx.from_numpy_array`) ;
6. Run de l'algorithme du Pagerank (`networkx.pagerank`);
7. Ranking des phrases dans l'ordre décroissant du score de Pagerank.

Dans l'algorithme original de *Textrank*, les étapes 3 et 4 n'utilisent pas d'approches de plongements lexicaux (word embeddings). Nous proposons donc l'implémentation d'une approche originale de *Textrank* exploitant les plongements lexicaux. 

Testez votre approche sur n'importe quel document et comparez là avec *gensim.summarization.summarizer*. 
Vous pouvez également tenter de résumer une publication scientifique, et comparer votre résumé avec *l'abstract* humain qui en est le résumé produit par les auteurs. Pour comparer deux résumés, vous pouvez considérer la métrique *Rouge* : https://pypi.org/project/rouge/


**[Mihalcea et Tarau]** Mihalcea, R., & Tarau, P. (2004). Textrank: Bringing order into text. In Proceedings of the 2004 conference on empirical methods in natural language processing (pp. 404-411).

**[Brin et Page]** Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual web search engine. Computer networks and ISDN systems, 30(1-7), 107-117.

