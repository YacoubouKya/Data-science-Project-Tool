# modules/modeling.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from typing import Tuple, Any
from modules.utils import helpers
from modules.utils.metrics import classification_metrics, regression_metrics
from math import isfinite

def _format_metrics(d: dict, decimals=3):
    """Arrondit les valeurs numériques du dict pour l'affichage."""
    out = {}
    for k, v in d.items():
        try:
            fv = float(v)
            if not isfinite(fv):
                out[k] = v
            else:
                out[k] = round(fv, decimals)
        except Exception:
            out[k] = v
    return out

def run_modeling(df: pd.DataFrame) -> dict:
    st.subheader("⚡ Modélisation interactive")

    cols = df.columns.tolist()
    target = st.selectbox("Choisir la variable cible", [""] + cols)
    if not target:
        st.info("Sélectionne une variable cible pour lancer l'entraînement.")
        st.stop()

    X = df.drop(columns=[target])
    y = df[target]

    task = st.selectbox("Type de tâche", ["auto", "classification", "regression"], index=0)
    if task == "auto":
        if y.dtype == "O" or (y.nunique() <= 20 and y.nunique()/len(y) < 0.1):
            task = "classification"
        else:
            task = "regression"
    st.write("👉 Tâche détectée :", task)

    test_size = st.slider("Taille test (%)", 5, 50, 20) / 100.0
    random_state = int(st.number_input("Seed aléatoire", value=42))

    model_choice = st.selectbox("Choisir un modèle", ["auto", "random_forest", "gradient_boosting", "linear/logistic"])
    do_scale = st.checkbox("⚙️ Standardiser les numériques", value=True)

    # Hyperparamètres exposés
    if model_choice in ["random_forest", "auto"]:
        rf_n_estimators = int(st.number_input("RF - n_estimators", 10, 1000, 100))
        rf_max_depth = int(st.number_input("RF - max_depth (0=>None)", 0, 50, 0))
    else:
        rf_n_estimators = 100; rf_max_depth = 0

    if model_choice in ["gradient_boosting", "auto"]:
        gb_n_estimators = int(st.number_input("GB - n_estimators", 10, 1000, 100))
        gb_max_depth = int(st.number_input("GB - max_depth", 1, 20, 3))
        gb_lr = float(st.number_input("GB - learning_rate", 0.01, 1.0, 0.1))
    else:
        gb_n_estimators = 100; gb_max_depth = 3; gb_lr = 0.1

    if st.button("🚀 Lancer l'entraînement"):
        # Préprocessing pipeline (construit sans boucle coûteuse)
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            X[cat_cols] = X[cat_cols].astype(str)

        num_steps = []
        if num_cols:
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if do_scale:
                num_steps.append(("scaler", StandardScaler()))

        cat_steps = []
        if cat_cols:
            cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]

        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline(num_steps), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline(cat_steps), cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

        # Choix du modèle (sans boucle)
        if model_choice == "auto":
            if task == "classification":
                model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=None if rf_max_depth==0 else rf_max_depth, random_state=random_state)
            else:
                model = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=None if rf_max_depth==0 else rf_max_depth, random_state=random_state)
        elif model_choice == "random_forest":
            model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=None if rf_max_depth==0 else rf_max_depth, random_state=random_state) if task=="classification" else RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=None if rf_max_depth==0 else rf_max_depth, random_state=random_state)
        elif model_choice == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=gb_n_estimators, max_depth=gb_max_depth, learning_rate=gb_lr, random_state=random_state) if task=="classification" else GradientBoostingRegressor(n_estimators=gb_n_estimators, max_depth=gb_max_depth, learning_rate=gb_lr, random_state=random_state)
        else:
            model = LogisticRegression(max_iter=1000) if task=="classification" else LinearRegression()

        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        # Split & train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        # Évaluation (metrics utilitaires)
        if task == "classification":
            metrics = classification_metrics(y_test, preds)
        else:
            metrics = regression_metrics(y_test, preds)

        metrics_display = _format_metrics(metrics, decimals=4)
        st.write("📊 **Metrics (test)** :")
        st.json(metrics_display)

        # Sauvegarde modèle et datasets
        helpers.ensure_dir("outputs/models")
        model_path = f"outputs/models/model_{target}.pkl"
        joblib.dump(pipe, model_path)
        st.success(f"✅ Modèle entraîné et sauvegardé : {model_path}")

        helpers.ensure_dir("outputs/data")
        X_train.assign(**{target: y_train}).to_csv(f"outputs/data/train_{target}.csv", index=False)
        X_test.assign(**{target: y_test}).to_csv(f"outputs/data/test_{target}.csv", index=False)

        # Stocker dans session_state pour reporting/evaluation
        st.session_state.update({
            "model": pipe,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "task": task,
            "y_pred": preds,
            "evaluation_metrics": pd.DataFrame([metrics_display])
        })

        # Feature importance si disponible (essayer d'extraire proprement)
        try:
            m = pipe.named_steps["model"]
            if hasattr(m, "feature_importances_") and len(num_cols) > 0:
                # get feature names from preprocessor
                try:
                    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
                except:
                    feature_names = num_cols + cat_cols
                fi = pd.Series(m.feature_importances_, index=feature_names).sort_values(ascending=False)
                st.session_state["feature_importance"] = fi
        except Exception:
            pass

        return {
            "pipeline": pipe,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "task": task
        }

    st.stop()