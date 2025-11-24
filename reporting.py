# modules/reporting.py

import streamlit as st
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

OUT_DIR = "outputs/reports"
os.makedirs(OUT_DIR, exist_ok=True)

def _img_to_base64(fig, width=600):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_str}" width="{width}"><br>'

def generate_report(session_state: dict):
    st.subheader("📝 Générer rapport consolidé")
    title_default = f"Rapport_consolidé_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    title = st.text_input("Titre du rapport", value=title_default)

    if st.button("📄 Créer rapport HTML"):
        html = [f"<html><head><meta charset='utf-8'><title>{title}</title></head><body style='font-family:sans-serif'>"]
        html.append(f"<h1 style='color:#1E3A5F'>{title}</h1>")

        # 1. Données brutes
        if "data" in session_state:
            df = session_state["data"]
            html.append(f"<h2 style='color:#1569C7'>1. Données brutes</h2>")
            html.append(f"<p>Dimensions : {df.shape[0]} × {df.shape[1]}</p>")
            html.append("<h4>Aperçu (5 premières lignes)</h4>")
            html.append(df.head(5).to_html(index=False))
            html.append("<h4>Résumé statistique (numérique)</h4>")
            html.append(df.describe().round(4).to_html())

            missing = df.isna().sum()
            html.append("<h4>Valeurs manquantes par colonne</h4>")
            html.append(missing.to_frame('missing').to_html())

            # Histogrammes : limiter le nombre d'images pour performance
            num_cols = df.select_dtypes(include='number').columns.tolist()
            to_plot = num_cols[:6]
            if to_plot:
                html.append("<h4>Histogrammes (exemples)</h4>")
                for col in to_plot:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col].dropna(), kde=True, ax=ax)
                    ax.set_title(col)
                    html.append(_img_to_base64(fig, width=400))

        # 2. Données préparées & log
        if "clean_data" in session_state:
            cdf = session_state["clean_data"]
            html.append(f"<h2 style='color:#1569C7'>2. Données préparées</h2>")
            html.append(f"<p>Dimensions : {cdf.shape[0]} × {cdf.shape[1]}</p>")
            html.append("<h4>Aperçu (5 premières lignes)</h4>")
            html.append(cdf.head(5).to_html(index=False))

            if "correction_log" in session_state:
                html.append("<h4>Log des corrections appliquées</h4>")
                html.append(session_state["correction_log"].to_html(index=False))

            # histo réduits
            num_cols_clean = cdf.select_dtypes(include='number').columns.tolist()[:6]
            if num_cols_clean:
                html.append("<h4>Histogrammes (préparés) - exemples</h4>")
                for col in num_cols_clean:
                    fig, ax = plt.subplots()
                    sns.histplot(cdf[col].dropna(), kde=True, ax=ax)
                    ax.set_title(col)
                    html.append(_img_to_base64(fig, width=400))

        # 3. Modèle
        if "model" in session_state:
            model_obj = session_state["model"]
            model_name = type(model_obj).__name__
            html.append(f"<h2 style='color:#1569C7'>3. Modèle entraîné</h2>")
            html.append(f"<p>Nom du modèle / pipeline : <b>{model_name}</b></p>")
            if all(k in session_state for k in ("X_train","X_test","y_train","y_test")):
                html.append("<ul>")
                html.append(f"<li>X_train : {session_state['X_train'].shape}</li>")
                html.append(f"<li>X_test : {session_state['X_test'].shape}</li>")
                html.append(f"<li>y_train : {session_state['y_train'].shape}</li>")
                html.append(f"<li>y_test : {session_state['y_test'].shape}</li>")
                html.append("</ul>")

            # feature importance (si fournie)
            if "feature_importance" in session_state:
                fi = session_state["feature_importance"]
                fig, ax = plt.subplots(figsize=(6,4))
                fi.plot(kind='bar', ax=ax)
                ax.set_ylabel("Importance")
                html.append(_img_to_base64(fig, width=600))

            # préd vs réel (si présents)
            if "y_pred" in session_state and "y_test" in session_state:
                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(session_state["y_test"], session_state["y_pred"], alpha=0.5)
                ax.plot([session_state["y_test"].min(), session_state["y_test"].max()],
                        [session_state["y_test"].min(), session_state["y_test"].max()], 'r--')
                ax.set_xlabel("Réel"); ax.set_ylabel("Prédiction")
                html.append(_img_to_base64(fig, width=600))

        # 4. Évaluation
        if "evaluation_metrics" in session_state:
            html.append(f"<h2 style='color:#1569C7'>4. Évaluation du modèle</h2>")
            em = session_state["evaluation_metrics"]
            # arrondir pour lisibilité
            try:
                em_display = em.round(4)
            except:
                em_display = em
            html.append(em_display.to_html(index=False))

        # Meta-info
        html.append("<h2 style='color:#1569C7'>5. Meta-info</h2>")
        html.append(f"<p>Date génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        if "data" in session_state:
            html.append(f"<p>Dimensions dataset initial : {session_state['data'].shape}</p>")
        if "clean_data" in session_state:
            html.append(f"<p>Dimensions dataset nettoyé : {session_state['clean_data'].shape}</p>")

        html.append("</body></html>")

        safe_title = title.replace(" ", "_")
        out_path = os.path.join(OUT_DIR, f"{safe_title}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))

        st.success(f"✅ Rapport généré : {out_path}")
        st.markdown(f"[Ouvrir le rapport]({out_path})")

        with open(out_path, "rb") as f:
            st.download_button(label="📥 Télécharger le rapport HTML", data=f, file_name=f"{safe_title}.html", mime="text/html", key=f"{safe_title}_download")