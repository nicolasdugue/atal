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


## L'approche TextRank

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

