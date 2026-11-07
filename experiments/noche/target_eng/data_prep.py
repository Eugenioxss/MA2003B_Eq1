"""
experiments/noche/target_eng/data_prep.py
=========================================
Carga del panel y precomputo/cache de features V4_spatial para los experimentos
de formulacion de target.
"""
import sys
import os
import time
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXP_DIR = os.path.join(ROOT, 'experiments')
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from common import load_panel, split, MODEL_STATIONS, HORIZONS
from planA_quickwins.features_planA import build_planA_features

CACHE_FILE = os.path.join(os.path.dirname(__file__), '_cache_v4.parquet')
FEATS_FILE = os.path.join(os.path.dirname(__file__), '_cache_features.json')

def get_v4_data(use_cache: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Devuelve (df_v4, feature_cols) con panel completo, categorica Estacion y targets."""
    import json
    if use_cache and os.path.exists(CACHE_FILE) and os.path.exists(FEATS_FILE):
        t0 = time.time()
        print(f"Cargando features V4 desde cache: {CACHE_FILE} ...")
        df = pd.read_parquet(CACHE_FILE)
        df['Estacion'] = df['Estacion'].astype('category')
        with open(FEATS_FILE, 'r') as f:
            feats = json.load(f)
        print(f"Features V4 cargados en {time.time()-t0:.2f}s ({df.shape[0]} filas, {len(feats)} features).")
        return df, feats

    print("Construyendo features V4_spatial desde panel...")
    t0 = time.time()
    panel = load_panel()
    df, feats = build_planA_features(panel, variant='V4')
    print(f"Features V4 construidos en {time.time()-t0:.2f}s.")

    # Guardar cache local para no recomputar en cada script
    print(f"Guardando cache en {CACHE_FILE}...")
    df.to_parquet(CACHE_FILE, index=False)
    with open(FEATS_FILE, 'w') as f:
        json.dump(feats, f)
    print("Cache guardado con exito.")
    return df, feats

if __name__ == '__main__':
    df, feats = get_v4_data(use_cache=True)
    tr, va, ts = split(df)
    print(f"Split listo: Train={len(tr):,}, Val={len(va):,}, Test={len(ts):,}")
