"""
ml_engine.py — Modul Kecerdasan Buatan
Model : Random Forest Regressor
Proyeksi 2021-2025: variabel iklim diproyeksikan via regresi linear historis
sehingga setiap tahun proyeksi punya nilai berbeda sesuai tren data.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats as sp_stats

from data_pipeline import (
    build_ml_features,
    FEATURE_COLS, TARGET_COL, TAHUN_COL, PROV_COL,
    SUMATERA_PROVINCES,
)

PROJECTION_YEARS = list(range(2021, 2026))
RANDOM_STATE     = 42


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Melatih model Random Forest …")
def train_model(df: pd.DataFrame):
    X_sc, y_sc, le, scaler_X, scaler_y, ml_cols = build_ml_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_sc, test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=300, max_depth=10,
        min_samples_split=4, min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_tr, y_tr)
    y_pred = scaler_y.inverse_transform(model.predict(X_te).reshape(-1,1)).ravel()
    y_true = scaler_y.inverse_transform(y_te.reshape(-1,1)).ravel()
    mape   = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    metrics = {
        "MAE" : round(float(mean_absolute_error(y_true, y_pred)), 0),
        "R²"  : round(float(r2_score(y_true, y_pred)), 4),
        "MAPE": round(float(mape), 2),
    }
    return model, le, scaler_X, scaler_y, ml_cols, metrics


# ─────────────────────────────────────────────────────────────────────────────
# PROYEKSI VARIABEL IKLIM via REGRESI LINEAR
# ─────────────────────────────────────────────────────────────────────────────
def project_climate_features(df: pd.DataFrame, province: str) -> dict:
    """
    Proyeksikan setiap fitur (Luas Panen, Curah hujan, Kelembapan, Suhu)
    untuk tahun 2021-2025 menggunakan regresi linear OLS atas data historis
    1993-2020 provinsi tersebut.

    Untuk Luas Panen: terapkan floor = 10% dari nilai minimum historis
    agar tidak proyeksi negatif/tidak wajar.

    Returns dict:
        {tahun: {feat: nilai, ...}, ...}
    """
    d = df[df[PROV_COL] == province].sort_values(TAHUN_COL)
    result = {}

    for feat in FEATURE_COLS:
        slope, intercept, r_val, p_val, _ = sp_stats.linregress(d[TAHUN_COL], d[feat])
        floor = d[feat].min() * 0.10  # batas bawah 10% dari minimum historis

        for yr in PROJECTION_YEARS:
            if yr not in result:
                result[yr] = {}
            val = intercept + slope * yr
            val = max(val, floor)          # hindari proyeksi terlalu ekstrem ke bawah
            result[yr][feat] = round(float(val), 3)

    # Simpan juga slope & r² untuk ditampilkan di UI
    trend_meta = {}
    for feat in FEATURE_COLS:
        slope, intercept, r_val, p_val, _ = sp_stats.linregress(d[TAHUN_COL], d[feat])
        trend_meta[feat] = {
            "slope"    : round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "r2"       : round(float(r_val**2), 4),
            "p_val"    : round(float(p_val), 4),
        }

    return result, trend_meta


# ─────────────────────────────────────────────────────────────────────────────
# PROYEKSI PRODUKSI 1 PROVINSI (pakai iklim hasil regresi)
# ─────────────────────────────────────────────────────────────────────────────
def project_province(
    df: pd.DataFrame,
    province: str,
    model, le, scaler_X, scaler_y, ml_cols: list,
) -> tuple[pd.DataFrame, dict]:
    """
    Proyeksi produksi 2021-2025 untuk satu provinsi.
    Variabel iklim per tahun diperoleh dari project_climate_features().

    Returns
    -------
    df_proj   : DataFrame [Tahun, Provinsi, Produksi, fitur iklim, is_projection]
    trend_meta: dict metadata tren per fitur
    """
    climate_proj, trend_meta = project_climate_features(df, province)

    try:
        prov_enc = int(le.transform([province])[0])
    except ValueError:
        prov_enc = 0

    rows = []
    for yr in PROJECTION_YEARS:
        feats = climate_proj[yr]
        feat_vals = [feats[c] for c in FEATURE_COLS] + [prov_enc, yr]
        X_row = np.array([feat_vals])
        X_sc  = scaler_X.transform(X_row)
        y_sc  = model.predict(X_sc)
        prod  = float(scaler_y.inverse_transform(y_sc.reshape(-1, 1))[0, 0])

        rows.append({
            TAHUN_COL        : yr,
            PROV_COL         : province,
            TARGET_COL       : max(round(prod, 0), 0),
            "Luas Panen"     : feats["Luas Panen"],
            "Curah hujan"    : feats["Curah hujan"],
            "Kelembapan"     : feats["Kelembapan"],
            "Suhu rata-rata" : feats["Suhu rata-rata"],
            "is_projection"  : True,
        })

    return pd.DataFrame(rows), trend_meta


# ─────────────────────────────────────────────────────────────────────────────
# PROYEKSI SEMUA PROVINSI (untuk peta)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def project_all_provinces(
    _model, _le, _scaler_X, _scaler_y,
    ml_cols_tuple, df_hash: str, _df: pd.DataFrame
) -> pd.DataFrame:
    ml_cols = list(ml_cols_tuple)
    frames  = []
    for prov in SUMATERA_PROVINCES:
        df_p, _ = project_province(_df, prov, _model, _le, _scaler_X, _scaler_y, ml_cols)
        frames.append(df_p)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# GABUNG HISTORIS + PROYEKSI
# ─────────────────────────────────────────────────────────────────────────────
def build_combined_timeseries(
    df_hist: pd.DataFrame,
    df_proj: pd.DataFrame,
    province: str,
) -> pd.DataFrame:
    h = df_hist[df_hist[PROV_COL] == province][[TAHUN_COL, PROV_COL, TARGET_COL]].copy()
    h["is_projection"] = False
    p = df_proj[[TAHUN_COL, PROV_COL, TARGET_COL, "is_projection"]].copy()
    return (
        pd.concat([h, p], ignore_index=True)
        .sort_values(TAHUN_COL)
        .reset_index(drop=True)
    )
