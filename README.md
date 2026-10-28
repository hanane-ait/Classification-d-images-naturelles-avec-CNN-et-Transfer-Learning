# Classification d’images naturelles avec CNN et Transfer Learning

## Présentation

Ce projet consiste à classifier des images du dataset **Natural Images** en utilisant des réseaux de neurones convolutifs (CNN) et le Transfer Learning avec MobileNetV2.

Trois approches sont comparées :
- CNN initial
- CNN amélioré
- Transfer Learning avec MobileNetV2

Les modèles sont comparés à partir de leurs performances sur les données de validation. Le modèle sélectionné est ensuite évalué sur les données de test.

Une application web développée avec Streamlit permet d'importer une image et d'obtenir une prédiction.

## Technologies utilisées

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Pillow
- Matplotlib

## Structure du projet

```text
NaturalImages-Classification/
├── TP_CNN_Natural_Images_avec_Transfer_Learning.ipynb
├── app.py
├── modele_images_selectionne.keras
├── class_names.json
├── requirements.txt
└── README.md
```

## Installation

Clonez le dépôt :

```bash
git clone https://github.com/VOTRE-NOM-UTILISATEUR/VOTRE-DEPOT.git
cd VOTRE-DEPOT
```

Créez un environnement virtuel :

```bash
python -m venv .venv
```

Sous Windows, activez-le :

```bash
.venv\Scripts\activate
```

Installez les dépendances :

```bash
pip install streamlit tensorflow numpy pandas pillow scikit-learn matplotlib
```

## Lancer l'application

Exécutez la commande suivante dans le dossier du projet :

```bash
python -m streamlit run app.py
```

Importez une image au format PNG, JPG ou JPEG pour obtenir la classe prédite et les scores associés.

## Évaluation

Les modèles sont comparés à l'aide des métriques de validation. Le modèle sélectionné est évalué sur les données de test avec notamment :

- Test loss
- Test accuracy
- Rapport de classification
- Matrice de confusion

