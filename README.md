# Clustering Kmeans de tweets

Dans le cadre de l'Unité d'Enseignement IF25 (data mining et réseaux sociaux), j'ai dévelopé un ensemble de codes permettant de faire du clustering de tweets à l'aide de l'algorithme Kmeans, un algorithme de clustering (classification non supervisée) non hiérarchique basé sur la supposition du nombre de communautés.
La classification se fait selon le choix du calculs de distances, de détermination des centres et des critères de convergence. Les caractéristiques du tweet utilisables pour le clustering sont le texte, on fait alors du clustering de profile, ou la date de post, on fait alors du clustering de tweets.

## Getting Started

Voici comment faire fonctionner le système de clustering, les dépendances et les codes à employer.

### Prerequisites

Le code est intégralement codé en python (python3) et nécessite donc l'installation de python3 : https://www.python.org/downloads/. Sous linux et mac, vous pouvez les installer via les commandes d'installation standard.


Toutes les dépendances appartiennent au repositories PyPi (pip), et ne nécessitent que l'installation de pip sur votre machine.

Sous linux : https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers

Sous windows et dans tous les cas : téléchargez https://bootstrap.pypa.io/get-pip.py
```
python get-pip.py
```

Sous mac 
```
sudo easy_install pip
```


Vous devez ensuite télécharger les dépendances via le fichier "requirements.txt" du code source :
```
pip install --upgrade -r requirements.txt
```
!!! Il se peut que vous ayez besoin de passer en super utilisateur.

### Installing

Vous avez simplement besoin d'installer le code source trouvé ici.

Via git :
```
git clone https://github.com/tnemelck/kmeans
```

## Running the tests

Pour vérifier que le code fonctionne bien, vous n'avez qu'à lancer le code main_demo :
```
python3 main_demo.py
```
Il est plus aisé de lancer tous les codes dans un IDE comme spyder pour profiter de meilleurs capacités d'affichage graphique.

### Break down into end to end tests

Les codes à lancer pour avoir des réusultats réels sont main_date.py et main_mots.py.
Il faudra penser à changer les paramètres internes de la fonction pour déterminer les répertoires de stockage ou les fonctions utilisées.
```
python3 main_date.py
```
```
python3 main_mots.py
```

### And coding style tests

Les fonctions qu'il peut être intéressant de connaître pour les implémenter ailleurs sont :
* dans kmeans.py : __init__, run, run_N, run_global, run_automated, cond_conv
* dans bdd.py : date_dir, concat_dir, drop_profile, bdd2bow

Je vous invite à vous intéresser à la doc de ces fonctions.

## Deployment

Les classifieurs implémentent le multi-processing, mais d'une manière qui n'est pas adapté du big data dù à l'enregistrement de variables temporaires de la librairie de multi-processing qui consomme trop d'espace mémoire.

## Built With

* Nos mains.

## Contributing

* Babiga Birregah
* Omar Jaafor
* Francois-Luc HAGHENBEEK

## Versioning

```
git pull 
```
devrait faire l'affaire.

## Authors

* **Clément Benigni** 

## License

Open source, faites en ce que vous voulez.
