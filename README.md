# Telco Customer Churn — Machine Learning & Pipeline de Préparation

Pipeline end-to-end d'exploration, de nettoyage ciblé et de modélisation prédictive sous Python pour anticiper l'attrition client (*churn*) dans le secteur des télécommunications.

---

## Contexte & Problématique Métier
Dans le secteur des télécoms, le coût d'acquisition d'un nouveau client est nettement supérieur à celui de sa fidélisation. L'objectif de ce projet est d'exploiter une base de 7 043 profils afin d'identifier les facteurs déclencheurs de résiliation et de déployer un pipeline de scoring prédictif.

---

## Architecture & Méthodologie

Le projet est structuré en deux volets complémentaires :

1. **Exploration & Feature Engineering (`churn_exploration.ipynb`) :**
   - **Diagnostic des types & Imputation :** Conversion de `TotalCharges` en valeur numérique et imputation par `0` pour les nouveaux clients (`tenure = 0`).
   - **Contrôle de distribution :** Détection d'aberrations sur `MonthlyCharges` par la méthode de l'écart interquartile (IQR).
   - **Encodage hybride :** 
     - Mapping binaire (`0`/`1`) pour les variables dichotomiques.
     - Encodage ordinal pour les durées d'engagement contractuel (`Contract`).
     - One-Hot Encoding (`get_dummies`) pour les modes de paiement et services afin de ne pas induire de hiérarchie artificielle.
   - **Discrétisation :** Création de segments d'ancienneté (*Nouveau*, *Etabli*, *Fidele*).

2. **Pipeline de Production & Modélisation (`telco_churn_pipeline.py`) :**
   - Encapsulation du flux de traitement dans une architecture orientée objet (`TelcoDataPipeline`).
   - **Stratification :** Découpage Train/Test (80/20) préservant le ratio naturel de la cible déséquilibrée (26,5 % de churn).
   - **Standardisation :** Normalisation via `StandardScaler` pour harmoniser l'échelle des métriques financières.
   - **Classification :** Entraînement d'un modèle de Régression Logistique avec évaluation des scores de précision, rappel et ROC-AUC.

---

## Résultats & Métriques (KPIs)
- **7 043** profils analysés avec 100 % de complétude post-traitement.
- **26,5 %** de taux de churn global conservé à l'entraînement et en phase de test grâce au split stratifié.
- Pipeline modulaire, réutilisable et prêt pour l'intégration en production.

---

## Stack Technique
- **Langage :** Python 3.10+
- **Manipulation de Données :** Pandas, NumPy
- **Machine Learning :** Scikit-Learn (`StandardScaler`, `train_test_split`, `LogisticRegression`, `metrics`)
- **Environnements :** Jupyter Notebook, Google Colab

---

## Installation & Exécution

Cloner le dépôt et installer les dépendances :
```bash
git clone [https://github.com/ton-profil/telco-churn-prediction.git](https://github.com/ton-profil/telco-churn-prediction.git)
cd telco-churn-prediction
pip install pandas numpy scikit-learn
