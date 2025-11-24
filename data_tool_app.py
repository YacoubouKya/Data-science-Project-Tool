# app.py
import streamlit as st
import pandas as pd
import sys
import os

# Ajouter le dossier modules au path pour Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(current_dir, 'modules')
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Imports des modules
try:
    import data_loader, eda, preprocessing, modeling, evaluation, reporting
except ImportError as e:
    st.error(f"❌ Erreur d'import des modules: {e}")
    st.info("Structure du répertoire:")
    st.code(f"Current dir: {current_dir}\nFiles: {os.listdir(current_dir)}")
    if os.path.exists(modules_path):
        st.code(f"Modules dir: {os.listdir(modules_path)}")
    st.stop()

from sklearn.model_selection import train_test_split

# ------------------------
# ⚙️ Configuration de la page
# ------------------------
st.set_page_config(
    page_title="Data Project Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------
# 🎨 CSS personnalisé
# ------------------------
st.markdown("""
<style>
.custom-header { position: fixed; top: 0; left: 0; width: 100%; height: 60px; background-color: #1E3A5F; color: white; display: flex; align-items: center; justify-content: space-between; padding: 0 40px; z-index: 9999; box-shadow: 0px 2px 5px rgba(0,0,0,0.3);}
.custom-header .logo { font-size: 22px; font-weight: bold; color: #FFD700; }
.custom-header .menu { display: flex; gap: 20px; }
.custom-header .menu a { color: white; text-decoration: none; font-weight: 500; font-family: 'Segoe UI', sans-serif; transition: color 0.3s;}
.custom-header .menu a:hover { color: #FFD700; }

.block-container { padding-top: 80px !important; }
.stApp { background-color: #1E3A5F; }
.block-container, .st-emotion-cache-18e3th9, .st-emotion-cache-1y4p8pa { background-color: transparent !important; }

h1, h2, h3, h4 { color: #FFD700; font-family: 'Segoe UI', sans-serif; }
p, span, label, div { color: #FFFFFF !important; font-family: 'Segoe UI', sans-serif; }

[data-testid="stSidebar"] { background-color: #1569C7 !important; color: yellow !important; }
[data-testid="stSidebar"] h1, h2, h3, label { color: yellow !important; }

.stButton>button { background-color: #FFD700; color: #1E3A5F; border-radius: 10px; padding: 10px 20px; border: none; font-weight: bold; }
.stButton>button:hover { background-color: #FFA500; color: white; }

[data-testid="stFileUploader"] { background-color: #FFD700 !important; border-radius: 10px; padding: 10px; }
[data-testid="stFileUploader"] label { color: #1E3A5F !important; font-weight: bold; }

div[role="radiogroup"] > label, .stSelectbox { background: #34495E !important; color: yellow !important; padding: 8px 15px; border-radius: 8px; margin: 3px 0; cursor: pointer; }
div[role="radiogroup"] > label:hover { background: #1ABC9C !important; }
</style>
""", unsafe_allow_html=True)

# Injection HTML du header
st.markdown("""
<div class="custom-header">
    <div class="logo">🐍 Data Project Tool</div>
    <div class="menu">
        <a href="#">About</a>
        <a href="#">Documentation</a>
        <a href="#">Community</a>
        <a href="#">Success Stories</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------
# 🎯 En-tête principal
# ------------------------
st.title("📊 Data Project Tool")
st.markdown("Bienvenue dans ton outil de projet data interactif 🚀")

# ------------------------
# 📌 Sidebar Navigation
# ------------------------
st.sidebar.title("📌 Navigation")
section = st.sidebar.radio(
    "Aller à :",
    ["📥 Chargement", "🔎 EDA", "🛠️ Prétraitement", "🤖 Modélisation", "📈 Évaluation", "📝 Reporting"]
)

# ------------------------
# Sections
# ------------------------
if section == "📥 Chargement":
    st.header("📥 Chargement des données")
    uploaded = st.file_uploader("Charger un fichier (CSV ou Excel)", type=["csv", "xlsx", "xls"])
    sep = ","; sheet = None
    if uploaded:
        if uploaded.name.lower().endswith(".csv"):
            sep = st.selectbox("Séparateur CSV", options=[",", ";", "\t"], index=0)
        elif uploaded.name.lower().endswith((".xls", ".xlsx")):
            xls = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Choisissez la feuille Excel", options=xls.sheet_names)
        df = data_loader.load_file(uploaded, sep=sep, sheet_name=sheet)
        if df is not None:
            st.session_state["data"] = df
            st.success("✅ Données chargées avec succès !")
            st.dataframe(df.head())

elif section == "🔎 EDA":
    st.header("🔎 Analyse exploratoire (EDA)")
    if "data" in st.session_state:
        eda.run_eda(st.session_state["data"])
    else:
        st.warning("⚠️ Chargez d'abord les données dans l'onglet Chargement.")

elif section == "🛠️ Prétraitement":
    st.header("🛠️ Prétraitement")
    if "data" in st.session_state:
        df = st.session_state["data"]
        profile = eda.generate_profile(df)
        if profile is None:
            st.warning("⚠️ Le profiling automatique n'est pas disponible. Détection d'anomalies limitée.")
            issues = []
        else:
            issues = preprocessing.detect_and_propose_corrections(profile, df)
        if issues:
            st.subheader("🚨 Anomalies détectées et corrections proposées")
            corrections_dict = {}
            for issue in issues:
                col = issue["colonne"]
                anomalies = ", ".join(issue["anomalies"])
                st.markdown(f"**Colonne : `{col}`**"); st.write(f"Anomalies : {anomalies}")
                choice = st.selectbox(f"Choisir correction pour `{col}`", ["Ne pas appliquer de correction"] + issue["propositions"], key=f"choice_{col}")
                corrections_dict[col] = choice
            if st.button("✅ Appliquer toutes les corrections sélectionnées"):
                valid_corrections = {col: corr for col, corr in corrections_dict.items() if corr != "Ne pas appliquer de correction"}
                if valid_corrections:
                    df_corrige, log_df = preprocessing.apply_corrections_with_log(df, valid_corrections)
                    st.session_state["clean_data"] = df_corrige
                    st.session_state["correction_log"] = log_df
                    st.success("✅ Toutes les corrections appliquées !")
                    st.subheader("📋 Tableau récapitulatif des corrections")
                    st.dataframe(log_df)
                    preprocessing.download_df(df_corrige, label="Télécharger la base corrigée", file_name="base_corrigee", file_format="excel")
                    preprocessing.download_df(log_df, label="Télécharger le log des corrections", file_name="log_corrections", file_format="excel")
                else:
                    st.info("Aucune correction sélectionnée à appliquer.")
        else:
            st.info("✅ Aucune anomalie détectée !")
    else:
        st.warning("⚠️ Chargez d'abord les données.")

elif section == "🤖 Modélisation":
    st.header("🤖 Modélisation")
    df_to_use = st.session_state.get("clean_data", st.session_state.get("data"))
    if df_to_use is not None:
        res = modeling.run_modeling(df_to_use)
        st.success("✅ Modèle entraîné et jeux train/test créés avec succès !")
        st.subheader("📌 Pipeline"); st.write(res["pipeline"])
        st.subheader("📌 Jeux de données")
        st.write(f"**X_train shape**: {res['X_train'].shape} | **X_test shape**: {res['X_test'].shape}")
        st.write(f"**y_train shape**: {res['y_train'].shape} | **y_test shape**: {res['y_test'].shape}")
        st.subheader("📌 Aperçu X_train"); st.dataframe(res["X_train"].head())
        st.subheader("📌 Aperçu y_train"); st.dataframe(res["y_train"].head())
        st.session_state.update({
            "model": res["pipeline"], "X_train": res["X_train"], "X_test": res["X_test"],
            "y_train": res["y_train"], "y_test": res["y_test"], "task_type": res["task"]
        })
    else:
        st.warning("⚠️ Chargez et/ou prétraitez d'abord les données.")

elif section == "📈 Évaluation":
    st.header("📈 Évaluation du modèle")
    if "model" in st.session_state:
        evaluation.run_evaluation(st.session_state["model"], st.session_state["X_test"], st.session_state["y_test"])
    else:
        st.warning("⚠️ Entraînez un modèle d'abord.")

elif section == "📝 Reporting":
    st.header("📝 Reporting")
    reporting.generate_report(st.session_state)