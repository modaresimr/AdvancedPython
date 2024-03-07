


# MinMaxScaler

MinMaxScaler est une technique de normalisation des données utilisée en apprentissage automatique et en statistiques. Elle met à l'échelle les caractéristiques dans une plage spécifiée, généralement entre 0 et 1, en appliquant la transformation suivante :

\[ X_{\text{normalisé}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}} \]

où \( X_{\text{min}} \) et \( X_{\text{max}} \) sont les valeurs minimale et maximale de la caractéristique \( X \), respectivement. Ce scaler préserve la forme de la distribution d'origine tout en normalisant la plage des données. Il est particulièrement utile lorsque les caractéristiques ont des échelles variables et aide à éviter que certains algorithmes soient dominés par des caractéristiques avec des échelles plus grandes.



## Utilisation de MinMaxScaler avec scikit-learn

Dans cette leçon, nous allons apprendre comment utiliser le MinMaxScaler de la bibliothèque scikit-learn pour mettre à l'échelle nos données.

### Étape 1 : Importer les bibliothèques nécessaires

```python
from sklearn.preprocessing import MinMaxScaler
```

### Étape 2 : Créer une instance de MinMaxScaler

```python
scaler = MinMaxScaler()
```

### Étape 3 : Adapter le scaler à nos données d'entraînement

```python
scaler.fit(X_train)
```

Ici, `X_train` représente notre ensemble de données d'entraînement.

### Étape 4 : Mettre à l'échelle les données d'entraînement 

```python
X_train_scaled = scaler.transform(X_train)
```

### Étape 5 : Mettre à l'échelle les données d'entraînement et de test

```
????
```

Maintenant, `X_train_scaled` et `X_test_scaled` contiennent nos données d'entraînement et de test mises à l'échelle, respectivement.

### Explication

Le MinMaxScaler est un outil qui met à l'échelle les données en transformant chaque valeur dans un intervalle spécifié, généralement entre 0 et 1. Cela aide à normaliser les données et à les rendre comparables. Dans notre cas, nous utilisons MinMaxScaler pour mettre à l'échelle nos données d'entraînement et de test, ce qui est une étape importante dans le processus de préparation des données pour l'apprentissage automatique.

N'oubliez pas d'adapter le scaler uniquement aux données d'entraînement et de transformer ensuite à la fois les données d'entraînement et de test. Cela garantit que les données de test ne sont pas utilisées pour adapter le scaler, ce qui évite tout biais dans l'évaluation de notre modèle.



# StandardScaler
StandardScaler est une autre technique de prétraitement des données disponible dans scikit-learn. Il standardise les caractéristiques en supprimant la moyenne et en mettant à l'échelle selon la variance unitaire. Ce processus transforme les données de telle manière qu'elles ont une moyenne de 0 et un écart type de 1. La standardisation est utile lorsque les caractéristiques de votre ensemble de données ont des échelles ou des unités différentes.

Voici la formule mathématique utilisée par StandardScaler pour transformer les données :

\[ X_{\text{standardisé}} = \frac{X - \mu}{\sigma} \]

Où :
- \( X \) est la valeur de la caractéristique d'origine,
- \( \mu \) est la moyenne de la caractéristique,
- \( \sigma \) est l'écart type de la caractéristique.

Dans scikit-learn, vous pouvez utiliser la classe `StandardScaler` pour standardiser vos données. Elle fonctionne de manière similaire à `MinMaxScaler`, où vous adaptez le scaler à vos données d'entraînement, puis transformez à la fois les données d'entraînement et de test en utilisant les paramètres de mise à l'échelle appris à partir des données d'entraînement.

Voici un exemple d'utilisation de `StandardScaler` dans scikit-learn :

```python
from sklearn.preprocessing import StandardScaler

# Créer une instance de StandardScaler
scaler = StandardScaler()

```

Ce processus garantit que vos données d'entraînement et de test sont mises à l'échelle de manière cohérente, ce qui est crucial pour de nombreux algorithmes d'apprentissage automatique pour fonctionner efficacement.



### Adapter le scaler à vos données d'entraînement
```
????
```
### Transformer les données d'entraînement et de test
```
????
```

##  StandardScaler vs MinMaxScaler
Le StandardScaler est souvent utilisé à la place du MinMaxScaler pour plusieurs raisons :

1. **Robustesse aux valeurs aberrantes :** Le StandardScaler est moins sensible à la présence de valeurs aberrantes par rapport au MinMaxScaler. Étant donné qu'il standardise les données en supprimant la moyenne et en les mettant à l'échelle pour avoir une variance unitaire, il est moins influencé par les valeurs extrêmes.

2. **Préservation de l'information :** Le StandardScaler préserve la forme de la distribution d'origine des données tout en les centrant sur zéro et en les mettant à l'échelle pour avoir une variance unitaire. Ceci peut être important lorsque la distribution des caractéristiques n'est pas nécessairement uniforme ou gaussienne.

3. **Interprétabilité :** Le StandardScaler produit des caractéristiques standardisées avec une moyenne de 0 et un écart type de 1. Cela facilite l'interprétation des coefficients dans les modèles linéaires, car ils reflètent le changement de la variable de sortie associé à un changement d'un écart type dans le prédicteur.

4. **Compatibilité avec les algorithmes :** Certains algorithmes, notamment ceux basés sur des calculs de distance (par exemple, les k-plus proches voisins, les machines à vecteurs de support), fonctionnent mieux avec des caractéristiques standardisées plutôt qu'avec des caractéristiques mises à l'échelle dans l'intervalle [0, 1].

Cependant, le choix entre StandardScaler et MinMaxScaler dépend finalement des caractéristiques spécifiques des données et des exigences de l'algorithme d'apprentissage automatique utilisé. Dans certains cas, MinMaxScaler peut être plus approprié, surtout lorsque les caractéristiques ont une plage bornée ou lorsque la préservation de l'interprétabilité de l'échelle d'origine est importante.



### Utilisez les données numériques du jeu de données
Utilisez uniquement les données numériques du jeu de données, appliquez le MinMaxScaler et le StandardScaler sur l'ensemble du jeu de données
Les colonnes numériques sont 'age', 'rest_bp', 'cholesterol', 'max_hr', 'st_depression'.

```
????
```

# utilisez les algorithm de Nearest Neighbors

```
???
```

# Créez une fonction qui prend un modèle, effectue des prédictions et imprime l'exactitude, la précision et le rappel sur une seule ligne.line

``` 
def evaluate(model):
    ????
```


# utilisez les algorithm d'Arbre de Décision.

```
?
```


En outre, utilisez d'autres modèles tels que Naive Bayes.

# utilisez les algorithmes Naive Bayes

``` 
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()

???
```

# utilisez les algorithmes RandomForest
``` 
from sklearn.ensemble import RandomForestClassifier

num_trees = 100
max_features = 'sqrt'

rf_model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)


???
```


# Comparez les précisions des algorithmes d'Arbre de Décision, de Nearest Neighbors, de Naive Bayes et de RandomForest.


``` 
???
```