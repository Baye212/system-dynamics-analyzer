# 🔬 Analyse Qualitative de Systèmes Dynamiques

Outil pédagogique interactif pour l'analyse complète des systèmes dynamiques linéaires et non-linéaires.

## 📋 Description

Cette application permet d'analyser qualitativement des systèmes dynamiques en 2D à travers plusieurs étapes :

1. **Analyse des valeurs propres** - Classification et nature des points fixes
2. **Vecteurs propres et sous-espaces** - Décomposition en sous-espaces stables, instables et centre
3. **Isoclines orientées** - Visualisation des régions de directions du flux
4. **Analyse par quadrant** - Tableau détaillé du comportement dans chaque quadrant
5. **Portrait de phase final** - Synthèse complète avec trajectoires et analyse qualitative

## 🚀 Déploiement

### Streamlit Cloud (Recommandé)

1. Allez sur [share.streamlit.io](https://share.streamlit.io)
2. Connectez-vous avec votre compte GitHub
3. Sélectionnez le dépôt `AZ473/system-dynamics-analyzer`
4. Le fichier principal est `app.py`
5. Cliquez sur "Deploy"

L'application sera accessible via une URL publique.

### Installation locale

```bash
# Cloner le dépôt
git clone https://github.com/AZ473/system-dynamics-analyzer.git
cd system-dynamics-analyzer

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

## 📦 Dépendances

- streamlit >= 1.26
- numpy >= 1.24
- matplotlib >= 3.7
- scipy >= 1.10

## 🎯 Fonctionnalités

- **Systèmes linéaires** : Analyse via matrice A (2x2)
- **Systèmes non-linéaires** : Analyse via fonctions f₁(x₁,x₂) et f₂(x₁,x₂)
- **Exemples prédéfinis** : Nœud stable, selle, centre, spirale stable
- **Génération de rapports PDF** : Export des analyses complètes

## 📁 Structure du projet

```
system-dynamics-analyzer/
├── app.py                 # Application Streamlit principale
├── main_interface.py      # Interface en ligne de commande
├── src/                   # Modules d'analyse
│   ├── dynamics_analyzer.py
│   ├── linear_analyzer.py
│   ├── nonlinear_analyzer.py
│   ├── phase_plotter.py
│   └── stability.py
├── examples/              # Exemples de systèmes
├── tests/                 # Tests unitaires
└── requirements.txt       # Dépendances Python
```

## 📝 Utilisation

1. Choisissez le type de système (Linéaire, Non-linéaire, ou Exemple prédéfini)
2. Entrez les paramètres du système
3. Cliquez sur "Analyser"
4. Explorez les résultats étape par étape
5. Téléchargez le rapport PDF si nécessaire

## ℹ️ À propos

Cette application a été conçue par **Mor DIOUF**, étudiant en **Master 2 Mathématiques Pures** à l'Université Assane Seck de Ziguinchor.

Elle est développée pour les étudiants de l'Université Assane Seck de Ziguinchor, ainsi que pour toute personne souhaitant apprendre les systèmes dynamiques.

L'application suit la méthodologie du professeur pour l'analyse des systèmes dynamiques :

- tracé des portraits de phase,
- classification des points fixes,
- calcul des sous-espaces propres,
- utilisation des isoclines,
- résolution des exercices avec une démarche pédagogique.

L'outil reprend pas à pas la méthode du cours pour proposer une analyse complète et structurée des systèmes dynamiques en 2D.

## 👤 Auteur

Mor DIOUF

## 📄 Licence

Ce projet est sous licence libre pour usage éducatif.
