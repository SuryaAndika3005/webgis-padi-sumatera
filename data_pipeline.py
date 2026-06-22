"""
data_pipeline.py  —  Modul Data & Spasial
Web GIS: Prediksi Produksi Padi Regional Sumatera Berdasarkan Faktor Iklim
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTA GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH     = "Data_Tanaman_Padi_Sumatera_version_1.csv"
GEOJSON_PATH = "38 Provinsi Indonesia - Provinsi.json"

SUMATERA_PROVINCES = [
    "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau",
    "Jambi", "Sumatera Selatan", "Bengkulu", "Lampung",
]

FEATURE_COLS = ["Luas Panen", "Curah hujan", "Kelembapan", "Suhu rata-rata"]
TARGET_COL   = "Produksi"
TAHUN_COL    = "Tahun"
PROV_COL     = "Provinsi"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPROCESS CSV
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat dataset …")
def load_csv(path: str = CSV_PATH) -> pd.DataFrame:
    """
    Muat CSV, lakukan Mean-Imputation outlier (IQR) pada fitur iklim,
    kembalikan DataFrame bersih (224 baris × 7 kolom).
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df[PROV_COL] = df[PROV_COL].str.strip()

    # Pastikan tipe numerik
    for col in FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Mean Imputation: outlier IQR → rata-rata non-outlier per provinsi
    climate_cols = ["Curah hujan", "Kelembapan", "Suhu rata-rata"]
    for col in climate_cols:
        for prov in df[PROV_COL].unique():
            mask   = df[PROV_COL] == prov
            series = df.loc[mask, col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            ok_mean = series[(series >= lo) & (series <= hi)].mean()
            outlier = mask & ((df[col] < lo) | (df[col] > hi) | df[col].isna())
            df.loc[outlier, col] = ok_mean

    # Isi NaN sisa (jika ada) dengan median global
    for col in FEATURE_COLS + [TARGET_COL]:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GEOJSON
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat GeoJSON …")
def load_geojson(path: str = GEOJSON_PATH) -> gpd.GeoDataFrame:
    """
    Muat GeoJSON 38 provinsi, filter ke 8 provinsi Sumatera.
    Kolom nama provinsi distandarisasi menjadi 'Provinsi'.
    """
    gdf = gpd.read_file(path)

    # GeoJSON ini memakai kolom 'PROVINSI' (huruf kapital semua)
    if "PROVINSI" in gdf.columns:
        gdf = gdf.rename(columns={"PROVINSI": PROV_COL})
    elif "Provinsi" not in gdf.columns:
        # Fallback: cari kolom string yang cocok
        for c in gdf.columns:
            if gdf[c].dtype == object and gdf[c].str.contains(
                "Aceh|Sumatera|Riau|Jambi|Bengkulu|Lampung", na=False
            ).any():
                gdf = gdf.rename(columns={c: PROV_COL})
                break

    gdf[PROV_COL] = gdf[PROV_COL].str.strip()
    gdf = gdf[gdf[PROV_COL].isin(SUMATERA_PROVINCES)].copy()
    gdf = gdf.reset_index(drop=True)

    # Pastikan CRS WGS-84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# BUILD ML FEATURES  (dipanggil oleh ml_engine.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_ml_features(df: pd.DataFrame):
    """
    Kembalikan tuple:
      X_scaled  – np.ndarray fitur [0,1]
      y_scaled  – np.ndarray target [0,1]
      le        – LabelEncoder Provinsi
      scaler_X  – MinMaxScaler fitur
      scaler_y  – MinMaxScaler target
      ml_cols   – list nama kolom fitur (urutan penting!)
    """
    df_enc = df.copy()

    le = LabelEncoder()
    df_enc["Provinsi_enc"] = le.fit_transform(df_enc[PROV_COL])

    ml_cols = FEATURE_COLS + ["Provinsi_enc", TAHUN_COL]

    X = df_enc[ml_cols].values
    y = df_enc[TARGET_COL].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y).ravel()

    return X_scaled, y_scaled, le, scaler_X, scaler_y, ml_cols


# ─────────────────────────────────────────────────────────────────────────────
# MERGE  (dipanggil oleh map_visualizer.py & app.py)
# ─────────────────────────────────────────────────────────────────────────────
def merge_year_data(
    gdf: gpd.GeoDataFrame,
    df: pd.DataFrame,
    year: int,
) -> gpd.GeoDataFrame:
    """
    Left-join GeoDataFrame dengan baris DataFrame untuk tahun tertentu.
    df bisa berisi data historis MAUPUN proyeksi (struktur kolom sama).
    """
    cols_keep = [PROV_COL, TARGET_COL, "Curah hujan", "Kelembapan",
                 "Suhu rata-rata", "Luas Panen"]
    df_year = df[df[TAHUN_COL] == year][cols_keep].copy()
    merged  = gdf.merge(df_year, on=PROV_COL, how="left")
    return merged
