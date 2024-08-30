
# Optimisation contextuelle d'un modèle de langage large 

Les modèles de langage ce sont des modèles d’IA ou des modèles de fondation conçus pour comprendre et générer le langage naturel, ils peuvent assuré plusieurs taches, la génération de textes, d’images, de vocaux, la traduction, etc. 

Ils sont basés sur de vastes ensembles de données et le plus connu est sans doute ChatGPT de OpenAI. Toutefois, il en existe de nombreux autres comme BERT de Google, Llama de Meta, BLOOM de Hugging Face, Falcon de Technology Innovation Institute et Mixtral de Mistral AI.


Le modèle sur le quel ce base ce travail est le modèle Mixtral sur la tache génération de textes pour des questions spécialisées.

Installer les dépendances requises : pip install -r requirements.txt

# Dépendances: 

Le projet nécessite plusieurs bibliothèques, tels que:

Transformers et PEFT essentielles pour l'utilisation du modèle de langage large.
Datasets pour le chargement des données depuis HF.
BeautifulSoup4 pour le scrapping

Voir requirements.txt pour la liste complète.

# Structure du Projet 

preprocessing.py : Gère les tâches de prétraitement des données collectées et utlisées pour le réglage fin. 

load_model.py : Charge et initialise les paramétres du modèle de fondation utilisé. 

fine_tuning.py : Effectue le fine-tuning du modèle pré-entraînés. 

inférence.py : Réalise l'inférence du modèle. 

scrapping_PMC.py et Scrapping_ArXiv.py : Scripts pour scraper les articles de PMC pour des données médicales et ArXiv pour des pub scientifiques sur les LLM.

Evaluation.py : Un exemple de scriptis sur l'évaluation d'une réponse par rapport à une réponse correcte. 

Combined_DataFrame.py : Fusionne et gère les données de diverses sources. 

Load_Data.py : Charge les données collectées (All_collected_data.xlsx) pour le traitement et l'analyse.

# utilisation


## Installation


```bash
  pip install -r requirements.txt 
```


Prétraitement des Données : 

```bash
preprocessing.py 
python preprocessing.py
```

Chargement des Modèles : 
```bash
load_model.py. 
python load_model.py
```
Fine-Tuning :
```bash
fine_tuning.py. 
python fine_tuning.py
```

Inférence et Évaluation : 

```bash
inférence.py 
Evaluation.py. 
python inférence.py
python Evaluation.py
``` 
