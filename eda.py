# eda.py
# modules/eda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from ydata_profiling import ProfileReport

def generate_profile(df: pd.DataFrame):
    """
    Génère un rapport de profiling avec ydata-profiling.
    Nécessite Python 3.11 (configuré via runtime.txt).
    """
    profile = ProfileReport(df, title="Profiling EDA", minimal=True)
    return profile

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
        st.success("✅ Rapport de profiling généré.")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("👁️ Afficher le rapport"):
                st.session_state.show_report = True
        with col2:
            if st.button("🙈 Masquer le rapport"):
                st.session_state.show_report = False
        with col3:
            with open("profiling_report.html", "rb") as f:
                st.download_button(label="💾 Télécharger le rapport HTML", data=f, file_name="profiling_report.html", mime="text/html")

        if st.session_state.show_report:
            with open("profiling_report.html", "r", encoding="utf-8") as f:
                report_html = f.read()
            st.components.v1.html(report_html, height=800, scrolling=True)

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