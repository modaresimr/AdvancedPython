# Initiation au Deep Learning avec Google Colab 

Source: https://moov.ai/fr/blog/deep-learning-avec-google-colab

Le Deep Learning et la science des données, deux sujets à la mode, qui sont sur toutes les langues! Vous aimeriez vous initier, mais ne savez pas comment configurer un environnement de développement Python sur votre ordinateur pour vos premiers projets.

Dans cet article, je vous présenterai les nombreux avantages d’un outil Cloud, simple, gratuit et adapté à la science des données : Google Colaboratory. Cet outil permet de développer des applications en Deep Learning en Python en un éclair. Pour commencer, il vous suffit simplement d’avoir un compte Gmail.

## Qu’est-ce que Google Colaboratory et quels en sont les avantages ?

Google Colaboratory ou Colab, un outil Google simple et gratuit pour vous initier au Deep Learning ou collaborer avec vos collègues sur des projets en science des données.

Colab permet :

-   D’améliorer vos compétences de codage en langage de programmation Python.  
    
-   De développer des applications en Deep Learning en utilisant des bibliothèques Python populaires telles que Keras, TensorFlow, PyTorch et OpenCV.  
    
-   D’utiliser un environnement de développement (Jupyter Notebook) qui ne nécessite aucune configuration.

Mais la fonctionnalité qui distingue Colab des autres services est l’accès à un  **processeur graphique GPU, totalement gratuitement**! Des informations détaillées sur le service sont disponibles sur la page  [FAQ de Colab](https://research.google.com/colaboratory/faq.html).

Comme son nom l’indique, Google Colaboratory s’accompagne du terme « collaboration ». En fait, Colab exploite les mêmes fonctionnalités de collaboration des autres éléments de la G Suite : Sheet, Slide, Docs, etc. Il fonctionne sur les serveurs Google et vous n’avez rien à installer.

De plus, les documents Colab (Jupyter Notebook) sont enregistrés directement votre compte Google Drive.

## Guide pas à pas pour activer Google Colab et développer votre premier modèle en Deep Learning

### Étape 1 – Créer un nouveau dossier sur Google Drive

Dans un premier temps, connectez-vous à votre compte Gmail (ou G Suite) puis rendez-vous dans l’application Google Drive et créez un nouveau dossier. Dans cet exemple, j’ai créé un dossier nommé « app » dans mon Google Drive. Vous pouvez bien sûr utiliser un nom différent.

![](https://moov.ai/wp-content/uploads/2019/01/Picture1.png)

Une fois le dossier créé, vous devriez obtenir un écran similaire à celui-ci :

![Dossier app dans My Drive](https://moov.ai/wp-content/uploads/2019/01/my-drive-folder.png)

Dossier « app » créé et vide.

### Étape 2 : Créer un nouveau fichier Colab

Dans votre nouveau dossier, faites un clic droit avec votre souris puis sélectionnez  **More > Colaboratory**.

![](https://moov.ai/wp-content/uploads/2019/01/creer-nouveau-fichier-colab.png)

  

Une fois dans le nouveau fichier, vous pouvez le renommer en cliquant sur le nom en haut du document.

![](https://moov.ai/wp-content/uploads/2019/01/renommer-document-colab.png)

### Étape 3 : Paramétrage du GPU gratuit (!)

Pour configurer le GPU (processeur graphique), il suffit de cliquer sur  **Edit > Notebook**  settings et  **sélectionner GPU**  comme accélérateur matériel.

![](https://moov.ai/wp-content/uploads/2019/01/colab-notebook-setting.png)

### Étape 4 : Exécuter du code Python de base

Nous pouvons dès maintenant commencer à utiliser Colab.

![](https://moov.ai/wp-content/uploads/2019/01/commencer-a-utiliser-colab.png)

À titre d’exemple, je vais exécuter quelques lignes de code du  [tutoriel Python Numpy](http://cs231n.github.io/python-numpy-tutorial/). Numpy est une librairie Python populaire en science des données utilisée pour des calculs mathématiques.

Si vous ne connaissez pas encore Python, c’est le langage de programmation le plus populaire en Intelligence Artificielle. Pour vous initier à Python je vous recommande  [ce tutoriel simple](https://www.w3schools.com/python/).

![](https://moov.ai/wp-content/uploads/2019/01/tutoriel-simple-python-numpy.png)

Ça fonctionne comme prévu &#128578; Pour exécuter le code, il suffit de cliquer sur le bouton play à gauche de la ligne de code.


## Concepts de base de Pandas

Quelques concepts de Pandas et de Python très basiques pour commencer.

#### Importer le package pandas

```python
import pandas as pd
```

#### Créer un DataFrame simple

- Syntaxe : `pd.DataFrame({colonne1 : valeur1, colonne2 : valeur2, colonne3 : valeur3})`

Vous pouvez avoir n'importe quoi comme noms de colonnes et n'importe quoi comme valeurs.

La seule exigence est d'avoir toutes les listes de valeurs de longueur égale (toutes ont une longueur de 3 dans cet exemple).

Il existe de nombreuses façons de créer un dataframe et vous en verrez quelques autres pendant le cours. Toutes peuvent être vues documentées [ici](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).

```python
df = pd.DataFrame({'nom':['Bob','Jen','Tim'],
                   'age':[20,30,40],
                   'animal_de_compagnie':['chat', 'chien', 'oiseau']})
```

```python
df
```

#### Afficher les noms de colonnes et les valeurs d'index

L'index est l'un des concepts les plus importants en pandas.

Chaque dataframe a un seul index qui est toujours disponible sous la forme `df.index` et si vous n'en fournissez pas (comme nous ne l'avons pas fait pour ce dataframe) un nouvel index est automatiquement créé.

Les index définissent la manière d'accéder aux lignes du dataframe.

L'index le plus simple est l'index de plage mais il existe des index plus complexes comme l'index d'intervalle, l'index de date et l'index multiple.

Nous explorerons les indexes plus en profondeur au cours de cette leçon.

```python
print(df.columns)
print(df.index)
```

#### Sélectionner une colonne par nom de deux manières différentes

Ces deux méthodes sont équivalentes et peuvent être utilisées presque toujours de manière interchangeable.

La principale exception est lorsque le nom de la colonne contient des espaces. Si par exemple nous avions une colonne appelée "ventes hebdomadaires" nous devons utiliser `df['ventes hebdomadaires']` car `df.ventes hebdomadaires` est une erreur syntaxique.

```python
print(df['nom'])
print(df.nom)
```

#### Sélectionner plusieurs colonnes

Pour sélectionner plusieurs colonnes, nous utilisons `df[colonnes_a_selectionner]` où `colonnes_a_selectionner` sont les colonnes qui nous intéressent données sous forme d'une liste python simple. Le résultat sera un autre dataframe.

C'est l'équivalent de lister les noms de colonnes dans la partie `SELECT` d'une requête sql.

```python
df[['nom','animal_de_compagnie']]
```

#### Sélectionner une ligne par index

La sélection régulière des lignes se fait via son index. Lorsque vous utilisez des index de plage, vous pouvez accéder aux lignes en utilisant des indices entiers mais cela ne fonctionnera pas lorsque vous utilisez un index de date par exemple.

Nous pouvons toujours accéder à n'importe quelle ligne du dataframe en utilisant `.iloc[i]` pour un certain entier i.

Le résultat est un objet série à partir duquel nous pouvons accéder aux valeurs en utilisant l'indexation de colonne.

```python
df.iloc[0]
```

sélectionnez la dernière ligne
```python
df.iloc[-1]
```


# Télécharger un fichier dans Google Colab

Pour télécharger un fichier dans Google Colab, vous pouvez utiliser `wget` qui fournit des fonctions pour télécharger et afficher des fichiers dans un notebook Colab.

Dans Jupyter, l'ajout de `!` au début d'une ligne indique que vous exécutez une commande shell, contrairement à du code Python. Cela signifie que vous effectuez des opérations liées au système d'exploitation, telles que la navigation dans les répertoires, la suppression de fichiers, l'exécution de programmes externes, etc. C'est un moyen pratique d'interagir avec le système d'exploitation depuis Jupyter.

```jupyter
!wget https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

# Charger un fichier CSV dans un DataFrame pandas
Pour charger un fichier CSV dans un DataFrame pandas dans Google Colab, vous pouvez utiliser la fonction `read_csv` de la bibliothèque pandas. Cette fonction peut lire des fichiers CSV à partir d'un chemin de fichier local ou d'une URL.

```
import pandas as pd
data= pd.read_csv('titanic.csv')
data
```

# Afficher les premières lignes d'un DataFrame pandas

Pour afficher les premières lignes d'un DataFrame pandas dans Google Colab, vous pouvez utiliser la méthode `head()`.

```
data.head()
```

# Examiner le ensamble de données
Une grande partie de la plupart des projets de machine learning consiste à connaître vos données. L'API pandas fournit une fonction describe qui produit les statistiques suivantes sur chaque colonne du DataFrame :

- count, qui est le nombre de lignes dans cette colonne. Idéalement, count contient la même valeur pour chaque colonne.

- mean et std, qui contiennent la moyenne et l'écart-type des valeurs dans chaque colonne.

- min et max, qui contiennent les valeurs les plus basses et les plus élevées dans chaque colonne.

- 25 %, 50 %, 75 %, qui contiennent divers quantiles.


```
data.describe()
```

### Filtrage des DataFrames

Vous pouvez filtrer les données en fonction des colonnes et des valeurs dans le dataframe.

#### Filtrer les données pour les hommes

Il y a deux éléments importants ici :
- `data.sex=='male'` donnera un tableau booléen où True signifie que la ligne a une colonne appelée sex qui a la valeur 'male'. Ce tableau numpy est appelé prédicat.
- data[data.sex=='male'] renverra toutes les lignes pour lesquelles le prédicat est vrai.

Le résultat de ce filtre est un dataframe avec les mêmes colonnes que le dataframe d'entrée.

```python
data[data.sex=='male']
```

#### Filtrer les âges pour les hommes

Encore une fois, il y a deux parties importantes :
- `data.sex=='male'` est le prédicat comme précédemment
- `data.age` signifie prendre les valeurs de la colonne age, et `data.age[data.sex=='male']` signifie prendre tous les âges qui sont liés aux lignes masculines.

Le résultat est une série pandas **pas** un dataframe.

```python
data.age[data.sex=='male']
```

### Ajout de méthodes aux filtres

Une méthode est une fonction et est souvent utilisée lors de l'analyse des données dans Pandas. Il existe d'innombrables méthodes Pandas. Nous passerons en revue quelques-unes des plus basiques pour montrer comment vous pouvez utiliser les méthodes pour analyser rapidement vos données.

#### Combien d'hommes et de femmes étaient sur le Titanic ?

Le pipeline suit toujours le même chemin :
- Le prédicat est évalué
- Les données sont filtrées en fonction d'un prédicat
- Une valeur agrégée est calculée après le filtrage.

La méthode count compte simplement le nombre de cadres dans le dataframe.

```python
data.sex[data.sex=='male'].count()
```

```python
data.sex[data.sex=='female'].count()
```

#### Quel était le taux de survie des hommes adultes (age>=18)

Ici, nous combinons des prédicats en utilisant l'opérateur et (&).

Cet opérateur applique l'opération logique et entre les éléments aux positions correspondantes.

Par exemple :
- x = np.array([True, False, True, True])
- y = np.array([False, True, False, True])
- donnera x & y = np.array([True & False, False & True, True & False, True & True]).

Dans l'exemple suivant, nous utilisons le combinateur ou (|).

Vous pouvez combiner deux tableaux numpy booléens tant qu'ils ont la même forme en utilisant les opérateurs & et |.

Combiner de simples listes Python de cette manière ne fonctionne pas.

```python
data.survived[(data.sex=='male')&(data.age>=18)].mean()
```

#### Quel était le taux de survie des femmes et des enfants ?

La méthode mean est la même qu'AVERAGE en SQL.

```python
data.survived[(data.sex=='female')|(data.age<18)].mean()
```

#### Utiliser groupby pour comparer les taux de survie des hommes et des femmes

La méthode `groupby` est l'un des outils les plus importants que vous utiliserez dans votre travail quotidien.

Son paramètre d'entrée principal est soit une chaîne indiquant un nom de colonne, soit une liste de chaînes indiquant une liste de noms de colonnes.

Son résultat est un objet GroupBy qui est très similaire à un dataframe.

Le fonctionnement de groupby est le même que GROUPBY SQL.

Pour plus d'informations, consultez la [documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html).

```python
data.groupby('sex')['survived'].mean()
```

#### Créer un DataFrame avec groupby

```python
new = data.groupby(['sex','pclass'])['survived','age'].mean()
new
```




# Modifier des valeurs d'une colonne dans un DataFrame pandas
Mettez à l'échelle l'étiquette.
```
data["Fare"] = data["Fare"] / 100.0
data
```


# Ajouter une colonne dans un DataFrame pandas
```
data2=data.copy()
data2["test"] = data2["Fare"] / data2["Fare"].max()
data2["test2"] = 1
data2["test3"] = data2["test2"]-data2["test"]
data2
```



# yData-Profiling : Analyse exploratoire de données simplifiée

Le principal objectif de yData-Profiling est de proposer une expérience d'Analyse Exploratoire de Données (EDA) simplifiée et condensée, offrant une solution uniforme et rapide. Tout comme la fonctionnalité utile de la fonction `df.describe()` de pandas, yData-Profiling offre une analyse étendue d'un DataFrame, tout en permettant l'exportation de l'analyse des données dans différents formats tels que html et json.

Le package génère une analyse simplifiée et résumée d'un ensemble de données, couvrant à la fois les données en série temporelle et textuelles.


L'installation de yData-Profiling se fait via la commande pip :

```python
!pip install ydata-profiling
```

Pour le dataset Titanic, on peut visualiser l'analyse avec:

```python
from ydata_profiling import ProfileReport

ProfileReport(data)
```

Cette opération générera un rapport complet sur le jeu de données Titanic, comprenant des informations sur les valeurs manquantes, les statistiques descriptives, les distributions, les corrélation entre les variables, etc.











# Création de différents ensembles de données pour les survivants et les non-survivants
```
df_survivants = df[df['Survived'] == 1]
df_nonsurvivants = df[df['Survived'] == 0]
```
Ces lignes de code créent deux ensembles de données distincts : l'un pour les passagers qui ont survécu et l'autre pour ceux qui n'ont pas survécu.

# Troisième distribution pour le test d'hypothèse - Tarifs des survivants
```
dist_c = df_survivants['Fare'].dropna()
dist_d = df_nonsurvivants['Fare'].dropna()
```
Cette ligne de code sélectionne les tarifs des passagers survivants et non survivants.


# Création d'une variable catégorielle pour les âges
```
df['AgeCat'] = ''
df['AgeCat'].loc[(df['Age'] < 18)] = 'jeune'
df['AgeCat'].loc[(df['Age'] >= 18) & (df['Age'] < 56)] = 'mature'
df['AgeCat'].loc[(df['Age'] >= 56)] = 'senior'
```
Ces lignes de code ajoutent une nouvelle colonne 'AgeCat' à l'ensemble de données, en catégorisant les âges des passagers en jeunes, matures ou seniors.

# Création d'une variable catégorielle pour les tailles de famille
```
df['FamilySize'] = ''
df['FamilySize'].loc[(df['SibSp'] <= 2)] = 'petite'
df['FamilySize'].loc[(df['SibSp'] > 2) & (df['SibSp'] <= 5 )] = 'moyenne'
df['FamilySize'].loc[(df['SibSp'] > 5)] = 'grande'
```
Ces lignes de code ajoutent une nouvelle colonne 'FamilySize' à l'ensemble de données, en catégorisant la taille de la famille des passagers en petite, moyenne ou grande.

# Création d'une variable catégorielle pour déterminer si le passager est seul
```
df['IsAlone'] = ''
df['IsAlone'].loc[((df['SibSp'] + df['Parch']) > 0)] = 'non'
df['IsAlone'].loc[((df['SibSp'] + df['Parch']) == 0)] = 'oui'
```

Ces lignes de code ajoutent une nouvelle colonne 'IsAlone' à l'ensemble de données, pour indiquer si le passager est seul ou non.

# Création d'une variable catégorielle pour indiquer si le passager est un jeune/homme mûr/senior ou une jeune/femme mûre/senior

```
df['SexCat'] = ''
df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] <= 21)] = 'jeunehomme'
df['SexCat'].loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50)] = 'hommemûr'
df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] > 50)] = 'seniorhomme'
df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] <= 21)] = 'jeunefemme'
df['SexCat'].loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50)] = 'femmemûre'
df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] > 50)] = 'seniorfemme'
```

Ces lignes de code ajoutent une nouvelle colonne 'SexCat' à l'ensemble de données, pour indiquer si le passager est un jeune homme, un homme mûr, un homme senior, une jeune femme, une femme mûre ou une femme senior.

# [Exercice](e1.ipynb)

