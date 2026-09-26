import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)


class TelcoDataPipeline:
    def __init__(self, data_url: str):
        self.data_url = data_url
        self.scaler = StandardScaler()
        self.df = None

    def charger_donnees(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_url)
        return self.df

    def nettoyer_et_imputer(self) -> pd.DataFrame:
        # Conversion du type object en numérique
        self.df["TotalCharges"] = pd.to_numeric(self.df["TotalCharges"], errors="coerce")

        # Imputation logique : les nouveaux clients (tenure=0) n'ont pas encore été facturés
        self.df["TotalCharges"] = self.df["TotalCharges"].fillna(0)
        return self.df

    def engineer_features(self) -> pd.DataFrame:
        # Binarisation
        colonnes_binaires = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in colonnes_binaires:
            self.df[col] = self.df[col].map({"Yes": 1, "No": 0})

        # Cible binaire
        self.df["Churn"] = self.df["Churn"].map({"Yes": 1, "No": 0})

        # Encodage ordinal hiérarchique
        ordre_contrat = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        self.df["Contract_encoded"] = self.df["Contract"].map(ordre_contrat)

        # Discrétisation métier de l'ancienneté
        def segmenter_tenure(mois):
            if mois <= 12:
                return "Nouveau"
            elif mois <= 48:
                return "Etabli"
            return "Fidele"

        self.df["tenure_categorie"] = self.df["tenure"].apply(segmenter_tenure)

        # One-Hot Encoding des variables nominales
        cols_one_hot = ["InternetService", "PaymentMethod", "tenure_categorie"]
        self.df = pd.get_dummies(self.df, columns=cols_one_hot, drop_first=True)

        return self.df

    def preparer_matrices(self):
        colonnes_a_retirer = [
            "customerID", "Churn", "Contract", "gender",
            "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        features = [col for col in self.df.columns if col not in colonnes_a_retirer]

        X = self.df[features]
        y = self.df["Churn"]

        # Split stratifié pour maintenir le ratio de 26,5 % de churn
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        cols_numeriques = ["tenure", "MonthlyCharges", "TotalCharges"]
        X_train[cols_numeriques] = self.scaler.fit_transform(X_train[cols_numeriques])
        X_test[cols_numeriques] = self.scaler.transform(X_test[cols_numeriques])

        return X_train, X_test, y_train, y_test


def executer_entrainement():
    pipeline = TelcoDataPipeline(DATASET_URL)
    pipeline.charger_donnees()
    pipeline.nettoyer_et_imputer()
    pipeline.engineer_features()
    X_train, X_test, y_train, y_test = pipeline.preparer_matrices()

    modele = LogisticRegression(max_iter=1000, random_state=42)
    modele.fit(X_train, y_train)

    y_pred = modele.predict(X_test)
    y_prob = modele.predict_proba(X_test)[:, 1]

    print("=== RAPPORT D'ÉVALUATION DU MODÈLE ===")
    print(classification_report(y_test, y_pred))
    print(f"Score ROC-AUC : {roc_auc_score(y_test, y_prob):.3f}")


if __name__ == "__main__":
    executer_entrainement()
