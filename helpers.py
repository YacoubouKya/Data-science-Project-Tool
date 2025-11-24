# helpers.py
# modules/utils/helpers.py
import os
import joblib

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_model(obj, path):
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)
    return path