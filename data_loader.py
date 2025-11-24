# Chargement de fichiers
# data_loader.py

# modules/data_loader.py
import pandas as pd
import io
import streamlit as st

def load_file(uploaded_file, sep=",", sheet_name=None):
    """
    Lit un fichier CSV ou Excel envoyé via Streamlit file_uploader.
    Retourne un DataFrame.
    
    - sep : séparateur du CSV (par défaut ",")
    - sheet_name : nom ou index de la feuille Excel (par défaut None = première feuille)
    """
    if uploaded_file is None:
        return None
    
    filename = uploaded_file.name.lower()
    df = None

    try:
        if filename.endswith(".csv"):
            # lecture CSV avec séparateur paramétrable
            try:
                df = pd.read_csv(uploaded_file, sep=sep)
            except Exception:
                uploaded_file.seek(0)
                content = uploaded_file.read()
                df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='ignore')), sep=sep)

        elif filename.endswith((".xls", ".xlsx")):
            # lecture Excel avec choix de feuille
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

        else:
            st.error("Format de fichier non supporté. Veuillez charger un CSV ou Excel.")
            return None

        st.success(f"Données chargées — {df.shape[0]} lignes × {df.shape[1]} colonnes")

    except Exception as e:
        st.error(f"Erreur lecture fichier: {e}")
        raise

    return df