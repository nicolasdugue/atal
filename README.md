# TD - Word embeddings pour le résumé automatique

L'approche *Textrank* pour le résumé automatique est une baseline très efficace et très communément utilisée **[Mihalcea et Tarau]**. L'approche *Textrank* est une approche de résumé automatique dite **extractive** : il s'agit de résumer un document via **l'extraction** de phrases considérées comme caractéristiques du contenu du document. *Textrank* permet également d'extraire des mots-clés pour un document afin de l'indexer, mais nous concentrerons ici sur l'approche de résumé automatique.
 
Cette approche est basée sur l'algorithme du *Pagerank*, l'algorithme qui a notamment rendu célèbre le moteur de recherche Google **[Brin et Page]**. Cet algorithme s'applique sur un graphe, et la construction du graphe est donc l'une des premières étapes de l'approche *Textrank* que nous détaillons ci-après : 
1. Séparation du document en phrases ;
2. Nettoyage des phrases ;
3. **-** Représentation vectorielle des phrases avec **word2vec**
4. Calcul de la similarité cosine de chacune des paires de phrases : matrice de similarité ;
5. Création d'un graphe valuée en utilisant la matrice de similarité ;
6. Run de l'algorithme du Pagerank ;
7. Ranking des phrases dans l'ordre décroissant du score de Pagerank.

Dans l'algorithme original de *Textrank*, les étapes 3 et 4 n'utilisent pas d'approches de plongements lexicaux (word embeddings). Nous proposons donc l'implémentation d'une approche originale de *Textrank* exploitant les plongements lexicaux. Pour cela, vous pouvez utiliser des *embeddings* pré-appris : http://vectors.nlpl.eu/repository/.

Pour la mise en oeuvre des étapes 5 et 6, vous pouvez utiliser networkx (*import networkx as nx*), une librairie Python pour manipuler des graphes très complète et très haut niveau :
- *nx.from_numpy_array* permet de créer le graphe à partir de la matrice de similarité ;
- *nx.pageranl* permet de faire tourner l'algorithme du Pagerank.

Testez votre approche sur n'importe quel document et comparez là avec par exemple *gensim.summarization.summarizer*. Pour comparer deux résumés, vous pouvez considérer la métrique *Rouge* qui sera présentée lors des exposés étudiants le 15 novembre : https://pypi.org/project/rouge/

**[Mihalcea et Tarau]** Mihalcea, R., & Tarau, P. (2004). Textrank: Bringing order into text. In Proceedings of the 2004 conference on empirical methods in natural language processing (pp. 404-411).

**[Brin et Page]** Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual web search engine. Computer networks and ISDN systems, 30(1-7), 107-117.

# TD - Word embeddings : Procruste 

## Les fichiers

On considère deux corpus : 
- frWac, un corpus du web français ;
- gdn, le corpus du grand débat national.

Sur ces deux corpus, nous avons appris des word embeddings avec Skip-gram with Negative sampling. Ces embeddings sont de dimension 700, et ils ont été appris sur les versions lemmatisées de ces corpus, on trouve donc le lemme du mot ajouté à la fin du mot :
\_adv pour adverbe,\_v pour verbe, \_n pour nom, \_a pour adjectif. Par exemple, le nom *filet* est représenté par *gilet_n*.

Pour ce TD, nous fournissons trois fichiers d'embeddings pour chacun des corpus...

**Pour frWac**
- [frWac complet](frwac_common_sorted.zip)
- [frWac très filtré](frwac_common_sorted_Uberfiltered.txt)
- [frWac peu filtré](frwac_common_sorted_filtered.txt)

Pour **gdn**
- [gdn complet](gdn_common_sorted.zip)
- [gdn très filtré](gdn_common_sorted_Uberfiltered.txt)
- [gdn peu filtré](gdn_common_sorted_filtered.txt)

Les fichiers très filtrés contiennent des embeddings sur les mots les plus fréquents, quand les fichiers peu filtrés contiennent des embeddings sur un vocabulaire plus large, et le fichier complet contient un vocabulaire encore plus large : 
```
$ wc -l frwac_common_sorted.txt 
18114 frwac_common_sorted.txt
$ wc -l frwac_common_sorted_filtered.txt 
3105 frwac_common_sorted_filtered.txt
$ wc -l frwac_common_sorted_Uberfiltered.txt 
615 frwac_common_sorted_Uberfiltered.txt
```

## Comment les utiliser ?

Vous pouvez utiliser les fichiers complets comme des KeyedVector Gensim.

Exemple : 
```
>>> from gensim.models import KeyedVectors
>>> wv_from_text = KeyedVectors.load_word2vec_format('gdn_common_sorted.txt', binary=False)
>>> wv_from_text.most_similar("gilet_n")
[('jaune_a', 0.9214566946029663), ('jaune_n', 0.6313734650611877), ('enfiler_v', 0.5787583589553833), ('détonateur_n', 0.5656031370162964), ('jacquerie_n', 0.5498842000961304), ('pacifiste_a', 	0.5418269634246826), ('contestataire_n', 0.5362532734870911), ('étincelle_n', 0.5354418754577637), ('manif_n', 0.5342225432395935), ('chienlit_n', 0.5236349105834961)]
```
Sur le corpus du grand débat, le vecteur d'embedding le plus similaire selon la similarité cosinus du vecteur d'embedding du nom *gilet* est le vecteur de l'adjectif *jaune*.

**Faire quelques tests de similarité avec des exemples**

**Essayer également en utilisant des opérations d'addition et de soustraction :**
```
>>> wv_from_text.most_similar(positive=['lion_n'], negative=['chat_n'])
[('ravager_v', 0.4405561089515686), ('exterminer_v', 0.4333677589893341), ('occident_n', 0.42082324624061584), ('calotte_n', 0.41908228397369385), ('prolétariat_n', 0.41813957691192627), ('pangolin_n', 0.4122292697429657), ('patauger_v', 0.4081956744194031), ('entrailles_n', 0.40781670808792114), ('météorite_n', 0.40622982382774353), ('égocentrisme_n', 0.4057267904281616)]
```

**Donner sur Slack vos exemples les plus curieux !**

## Comment comparer les embeddings appris sur frWac et ceux appris du gdn ?

On utilise un procruste pour aligner les embeddings appris sur les deux corpus, i.e. pour les projeter dans le même espace qui doit être celui de frWac ou celui du GDN.

[Définition du Procruste](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)

On cherche une matrice de transformation sigma capable de plonger les embeddings de l'espace A dans l'espace des embeddings de B. Pour cela, on prend donc un échantillon de mots communs fréquents dans les deux espaces que l'on va aligner. Pour les aligner, on applique une SVD sur la matrice BA^t. On obtient ainsi une décomposition en U * Sigma * V^t, et la matrice sigma a pour valeur Sigma=np.matmul(U,V_t).

Ensuite, on peut ainsi comparer les matrices [Sigma * A] et [B] : si les deux espaces ont bien été alignés, les vecteurs d'un meme mot seront proches.

1. Avec numpy, utiliser loadtxt pour charger les matrices *gdn_common_sorted_Uberfiltered.txt* dans A et *frwac_common_sorted_Uberfiltered.txt* dans B.
```
A=np.loadtxt("EtudeEmbeddings/gdn_common_sorted_Uberfiltered.txt", usecols=range(1,701))
```

2. En utilisant *orthogonal_procrustes* de scipy.linalg, réaliser l'alignement procruste.

3. Calculer les distance entre [Sigma * A] et [B] en utilisant la fonction cosine de *scipy.spatial.distance*. Faire le plot d'un histogramme.

4. Maintenant, charger *gdn_common_sorted_filtered.txt* dans A et *frwac_common_sorted_filtered.txt* dans B puis comparer à nouveau [Sigma * A] et [B] en conservant le Sigma déjà appris. Qu'observe-t-on ?

5. Afficher les mots qui ont été les plus mal alignés. L'hypothèse est que ces mots sont les mots dont le sens a le plus changé dans les deux corpus.


# Liste des papiers à lire 

**TODO 15 Novembre** : 10mn de présentation + 10 mn de questions. 

[Exposés Atal](https://lite.framacalc.org/papiersalire)
