# eda.py
# modules/eda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import numpy as np

def generate_profile(df: pd.DataFrame):
    """
    Génère un profiling manuel simple (ydata-profiling incompatible Python 3.13).
    Retourne un dictionnaire avec les statistiques de base.
    """
    profile = {
        'variables': {}
    }
    
    for col in df.columns:
        col_info = {
            'type': str(df[col].dtype),
            'count': int(df[col].count()),
            'missing': int(df[col].isna().sum()),
            'missing_pct': float(df[col].isna().sum() / len(df) * 100),
            'unique': int(df[col].nunique()),
        }
        
        # Statistiques pour colonnes numériques
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'mean': float(df[col].mean()) if df[col].notna().any() else None,
                'std': float(df[col].std()) if df[col].notna().any() else None,
                'min': float(df[col].min()) if df[col].notna().any() else None,
                'max': float(df[col].max()) if df[col].notna().any() else None,
                'zeros': int((df[col] == 0).sum()),
                'infinite': int(np.isinf(df[col]).sum()) if df[col].notna().any() else 0,
            })
        
        profile['variables'][col] = col_info
    
    # Créer un objet simple avec méthode get_description
    class SimpleProfile:
        def __init__(self, data):
            self.data = data
        
        def get_description(self):
            return self.data
        
        def to_file(self, filename):
            st.warning("⚠️ Export HTML non disponible (ydata-profiling incompatible Python 3.13)")
    
    return SimpleProfile(profile)

def run_eda(df: pd.DataFrame):
    st.subheader("Aperçu général")
    st.write("Dimensions :", df.shape)
    st.dataframe(df.head())

    st.markdown("**Statistiques descriptives (numériques)**")
    st.dataframe(df.describe().T.round(4))

    # --------------------------
    # Rapport de profiling
    # --------------------------
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
    if "show_report" not in st.session_state:
        st.session_state.show_report = False

    if not st.session_state.report_generated:
        if st.button("📊 Générer le rapport de Profiling"):
            prof = generate_profile(df)
            prof.to_file("profiling_report.html")
            st.session_state.report_generated = True
            st.session_state.show_report = True

    if st.session_state.report_generated:
        st.info("ℹ️ Profiling manuel généré (ydata-profiling incompatible Python 3.13)")
        st.markdown("**Analyse des colonnes disponible dans la section Prétraitement**")

    # --------------------------
    # Histogrammes : sélection interactive (évite boucle coûteuse par défaut)
    # --------------------------
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        st.subheader("Histogrammes (sélectionner une variable ou afficher un échantillon)")
        col_choice = st.selectbox("Choisir une variable à afficher", ["--Tous (limité)-->"] + num_cols)
        if col_choice == "--Tous (limité)-->":
            # On propose un échantillon des premières 6 variables pour éviter surcharge
            to_plot = num_cols[:6]
        else:
            to_plot = [col_choice]

        for col in to_plot:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Histogramme de {col}")
            st.pyplot(fig)
            plt.close(fig)

    # --------------------------
    # Corrélation
    # --------------------------
    if "corr_generated" not in st.session_state:
        st.session_state.corr_generated = False
    if "show_corr" not in st.session_state:
        st.session_state.show_corr = False

    if not st.session_state.corr_generated:
        if st.button("🔗 Générer la matrice de corrélation"):
            st.session_state.corr_generated = True
            st.session_state.show_corr = True

    if st.session_state.corr_generated:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👁️ Afficher corrélation"):
                st.session_state.show_corr = True
        with col2:
            if st.button("🙈 Masquer corrélation"):
                st.session_state.show_corr = False

        if st.session_state.show_corr:
            st.subheader("Matrice de corrélation")
            corr = df.corr(numeric_only=True)
            # arrondir pour lisibilité
            corr_display = corr.round(3)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_display, annot=True, cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)
            plt.close(fig)