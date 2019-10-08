# TD word embeddings

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

**Faire quelques tests de similarité avec des exemples **

Essayer également en utilisant des opérations d'addition et de soustraction : 
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

2. En utilisant *svd* de scipy.linalg et *transpose* de numpy, réaliser l'alignement procruste.

3. Calculer les distance entre [Sigma * A] et [B] en utilisant la fonction cosine de *scipy.spatial.distance*. Faire le plot d'un histogramme.

4. Maintenant, charger *gdn_common_sorted_filtered.txt* dans A et *frwac_common_sorted_filtered.txt* dans B puis comparer à nouveau [Sigma * A] et [B] en conservant le Sigma déjà appris. Qu'observe-t-on ?

5. Afficher les mots qui ont été les plus mal alignés. L'hypothèse est que ces mots sont les mots dont le sens a le plus changé dans les deux corpus.
