# modules/preprocessing.py
# preprocessing.py

import pandas as pd
import streamlit as st
from io import BytesIO

# ------------------------
# Détection anomalies (utilise profile_report)
# ------------------------
def detect_and_propose_corrections(profile_report, df: pd.DataFrame):
    desc = profile_report.get_description()
    # compatibilité selon version
    vars_desc = desc.get("variables") if isinstance(desc, dict) else getattr(desc, "variables", {})
    results = []

    # Parcours du dictionnaire produit par profiling (faible coût, petite structure)
    for col, info in vars_desc.items():
        anomalies = []
        corrections = []

        n_missing = info.get("n_missing", 0) or 0
        if n_missing > 0:
            anomalies.append(f"{n_missing} valeurs manquantes")
            corrections.extend([
                "Imputer (moyenne)",
                "Imputer (médiane)",
                "Imputer (mode)",
                "Supprimer lignes",
                "Supprimer colonne"
            ])

        n_unique = info.get("n_unique", 0) or 0
        if n_unique == 1:
            anomalies.append("Colonne constante")
            corrections.append("Supprimer colonne")

        if len(df) > 0 and n_unique > 0.5 * len(df):
            anomalies.append("Cardinalité élevée")
            corrections.append("Encodage alternatif (hashing/target)")

        n_infinite = info.get("n_infinite", 0) or 0
        if n_infinite > 0:
            anomalies.append(f"{n_infinite} valeurs infinies")
            corrections.extend(["Remplacer par NaN + imputer", "Supprimer lignes"])

        if anomalies:
            results.append({
                "colonne": col,
                "anomalies": anomalies,
                "propositions": list(dict.fromkeys(corrections))  # garde l'ordre, dédoublonne
            })

    # Doublons purs (table-level) : tenter d'extraire depuis profile_report, fallback sur df
    duplicates_count = None
    if isinstance(desc, dict):
        # plusieurs chemins possibles selon version
        duplicates_count = desc.get("table", {}).get("n_duplicates") if isinstance(desc.get("table"), dict) else None
        if duplicates_count is None:
            duplicates_count = desc.get("dataset", {}).get("n_duplicates") if isinstance(desc.get("dataset"), dict) else None
        if duplicates_count is None:
            duplicates_count = desc.get("table", {}).get("n_duplicated") if isinstance(desc.get("table"), dict) else None

    if duplicates_count is None:
        duplicates_count = int(df.duplicated().sum())

    if duplicates_count and duplicates_count > 0:
        results.append({
            "colonne": "DOUBLONS",
            "anomalies": [f"{int(duplicates_count)} doublons purs détectés"],
            "propositions": ["Supprimer doublons purs", "Conserver"]
        })

    return results


# ------------------------
# Application correction avec suivi (optimisée)
# ------------------------
def apply_corrections_with_log(df: pd.DataFrame, corrections_dict: dict):
    """
    corrections_dict e.g. {'col1': 'Imputer (moyenne)', 'DOUBLONS': 'Supprimer doublons purs'}
    Retourne : df corrigé, log_df
    """
    log = []

    # Regrouper colonnes par type de correction pour appliquer vectorisé quand possible
    # Ex: imputer moyenne/mediane/mode pour plusieurs colonnes numériques
    # Construire des buckets
    buckets = {}
    for col, corr in corrections_dict.items():
        buckets.setdefault(corr, []).append(col)

    # Traitement pour doublons (dataset-level)
    if "Supprimer doublons purs" in buckets:
        cols = buckets.pop("Supprimer doublons purs")
        if "DOUBLONS" in cols:
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            log.append({
                "colonne": "DOUBLONS",
                "correction_appliquee": "Supprimer doublons purs",
                "nb_valeurs_modifiees": int(before - after)
            })

    # Imputation moyenne/mediane en vectorisé pour colonnes numériques uniquement
    # Moyenne
    if "Imputer (moyenne)" in buckets:
        cols = [c for c in buckets.pop("Imputer (moyenne)") if c in df.columns]
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            means = df[num_cols].mean()
            df[num_cols] = df[num_cols].fillna(means)
            for c in num_cols:
                log.append({"colonne": c, "correction_appliquee": "Imputer (moyenne)", "nb_valeurs_modifiees": int(df[c].isna().sum() == 0)})  # best-effort

        # non-numériques fallback: per-column mode
        cat_cols = [c for c in cols if c not in num_cols and c in df.columns]
        for c in cat_cols:
            mode_val = df[c].mode()[0] if not df[c].mode().empty else None
            before_missing = df[c].isna().sum()
            df[c] = df[c].fillna(mode_val)
            modified = int(before_missing - df[c].isna().sum())
            log.append({"colonne": c, "correction_appliquee": "Imputer (moyenne) (fallback mode)", "nb_valeurs_modifiees": modified})

    # Imputer mediane
    if "Imputer (médiane)" in buckets:
        cols = [c for c in buckets.pop("Imputer (médiane)") if c in df.columns]
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            meds = df[num_cols].median()
            before = df[num_cols].isna().sum()
            df[num_cols] = df[num_cols].fillna(meds)
            after = df[num_cols].isna().sum()
            for c in num_cols:
                log.append({"colonne": c, "correction_appliquee": "Imputer (médiane)", "nb_valeurs_modifiees": int(before[c] - after[c])})
        # categorical fallback
        cat_cols = [c for c in cols if c not in num_cols]
        for c in cat_cols:
            mode_val = df[c].mode()[0] if not df[c].mode().empty else None
            before_missing = df[c].isna().sum()
            df[c] = df[c].fillna(mode_val)
            log.append({"colonne": c, "correction_appliquee": "Imputer (médiane) (fallback mode)", "nb_valeurs_modifiees": int(before_missing - df[c].isna().sum())})

    # Imputer mode
    if "Imputer (mode)" in buckets:
        cols = [c for c in buckets.pop("Imputer (mode)") if c in df.columns]
        for c in cols:
            before_missing = df[c].isna().sum()
            mode_val = df[c].mode()[0] if not df[c].mode().empty else None
            df[c] = df[c].fillna(mode_val)
            log.append({"colonne": c, "correction_appliquee": "Imputer (mode)", "nb_valeurs_modifiees": int(before_missing - df[c].isna().sum())})

    # Supprimer lignes (per-column)
    if "Supprimer lignes" in buckets:
        cols = [c for c in buckets.pop("Supprimer lignes") if c in df.columns]
        for c in cols:
            before = len(df)
            df = df.dropna(subset=[c])
            after = len(df)
            log.append({"colonne": c, "correction_appliquee": "Supprimer lignes", "nb_valeurs_modifiees": int(before - after)})

    # Supprimer colonne
    if "Supprimer colonne" in buckets:
        cols = [c for c in buckets.pop("Supprimer colonne") if c in df.columns]
        for c in cols:
            df = df.drop(columns=[c])
            log.append({"colonne": c, "correction_appliquee": "Supprimer colonne", "nb_valeurs_modifiees": "colonne supprimée"})

    # Remplacer infinis + imputer
    if "Remplacer par NaN + imputer" in buckets:
        cols = [c for c in buckets.pop("Remplacer par NaN + imputer") if c in df.columns]
        for c in cols:
            before_neg = df[c].isna().sum()
            df[c] = df[c].replace([float("inf"), -float("inf")], pd.NA)
            if pd.api.types.is_numeric_dtype(df[c]):
                med = df[c].median()
                df[c] = df[c].fillna(med)
            else:
                mode_val = df[c].mode()[0] if not df[c].mode().empty else None
                df[c] = df[c].fillna(mode_val)
            log.append({"colonne": c, "correction_appliquee": "Remplacer par NaN + imputer", "nb_valeurs_modifiees": int(df[c].isna().sum() - before_neg)})

    # Encodage / autres => journaliser pour attention
    for corr, cols in list(buckets.items()):
        if corr.startswith("Encodage") or corr == "Ne pas corriger":
            for c in cols:
                log.append({"colonne": c, "correction_appliquee": corr, "nb_valeurs_modifiees": 0})
        else:
            for c in cols:
                # fallback safe: apply per-column
                if c in df.columns:
                    df_before = df[c].copy()
                    df = apply_correction(df, c, corr)
                    modified = int(df_before.ne(df[c]).sum()) if c in df.columns else "colonne supprimée"
                    log.append({"colonne": c, "correction_appliquee": corr, "nb_valeurs_modifiees": modified})
                else:
                    log.append({"colonne": c, "correction_appliquee": corr, "nb_valeurs_modifiees": "colonne absente"})

    log_df = pd.DataFrame(log)
    # formatage propre
    if "nb_valeurs_modifiees" in log_df.columns:
        # convertir en int quand possible
        def safe_int(x):
            try:
                return int(x)
            except:
                return x
        log_df["nb_valeurs_modifiees"] = log_df["nb_valeurs_modifiees"].apply(safe_int)

    return df, log_df


# ------------------------
# Fallback applicateur (colonne unique)
# ------------------------
def apply_correction(df: pd.DataFrame, col: str, correction: str) -> pd.DataFrame:
    if col not in df.columns and col != "DOUBLONS":
        return df

    if correction == "Imputer (moyenne)":
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            df[col] = df[col].fillna(mode_val)

    elif correction == "Imputer (médiane)":
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            df[col] = df[col].fillna(mode_val)

    elif correction == "Imputer (mode)":
        mode_val = df[col].mode()[0] if not df[col].mode().empty else None
        df[col] = df[col].fillna(mode_val)

    elif correction == "Supprimer lignes":
        df = df.dropna(subset=[col])

    elif correction == "Supprimer colonne":
        df = df.drop(columns=[col])

    elif correction == "Remplacer par NaN + imputer":
        df[col] = df[col].replace([float("inf"), -float("inf")], pd.NA)
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            df[col] = df[col].fillna(mode_val)

    elif correction.startswith("Encodage"):
        st.warning(f"⚠️ Encodage non encore implémenté pour {col}")

    return df


# ------------------------
# Télécharger base corrigée ou log
# ------------------------
def download_df(df: pd.DataFrame, label="Télécharger", file_name="data", file_format="csv"):
    if file_format == "csv":
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label=f"{label} (CSV)", data=csv_data, file_name=f"{file_name}.csv", mime="text/csv", key=f"{file_name}_csv")
    elif file_format == "excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        processed_data = output.getvalue()
        st.download_button(label=f"{label} (Excel)", data=processed_data, file_name=f"{file_name}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{file_name}_excel")
    else:
        st.error("Format non supporté. Choisissez 'csv' ou 'excel'.")