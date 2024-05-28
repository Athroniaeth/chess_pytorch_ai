# chess_pytorch_ai

## Introduction

Projet contenant mes tests sur la fabrication d'une IA pour jouer aux échecs, fonctionnant uniquement en Deep Learning,
en utilisant PyTorch. Le but est de voir ce qui est possible de faire en contraste avec les méthodes classiques de recherche
d'arbre de jeu (minimax, alpha-beta pruning, etc.) n'utilisant que des algorithmes de notation et heuristiques.

## Structure

```
├── src               <- Code source du projet
├── docs              <- Documentation du projet
│   └── static        <- Fichiers statiques du README.md
├── tests             <- Dossier contenants les tests logiciels
│   ├── units            <- Tests unitaires
│   └── integrations     <- Tests d'intégration
├── scripts           <- Scripts utiles pour le projet (pas de CI/CD)
├── ruff.toml         <- Fichier de configuration de Ruff
├── environment.yml   <- Fichier de configuration de l'environnement conda
```
## Installation
Ce projet nécessite d'avoir **conda** d'installé. Pour installer les dépendances, il suffit de lancer la commande suivante :

```bash
conda env create -f environment.yml
```

Vous pouvez mettre à jour l'environnement avec la commande suivante :

```bash
conda env update -f environment.yml
```

## Utilisation
Ce projet utilise `typer` pour créer une interface en ligne de commande. Pour lancer l'aide aux commandes, il suffit de lancer
la commande suivante :

```bash
python src
```

## Documentation

Ce projet est documenté en utilisant `mkdocs`. Pour lancer la documentation, il suffit de lancer la commande suivante :

```bash
mkdocs serve
```

Et de se rendre à l'adresse [`http://localhost:8000`](http://localhost:8000) pour consulter la documentation.
