"""
map_visualizer.py  —  Modul Peta Folium
Web GIS: Prediksi Produksi Padi Regional Sumatera Berdasarkan Faktor Iklim

Fungsi utama : render_choropleth_map()
- Choropleth berdasarkan kolom Produksi
- Tooltip  : nama provinsi + produksi (hover)
- Popup    : rincian suhu, curah hujan, kelembapan (klik)
"""

import folium
import geopandas as gpd
import numpy as np
import branca.colormap as cm
from folium.features import GeoJsonTooltip, GeoJsonPopup

from data_pipeline import TARGET_COL, PROV_COL

# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTA VISUAL
# ─────────────────────────────────────────────────────────────────────────────
SUMATERA_CENTER  = [-0.8, 101.5]
DEFAULT_ZOOM     = 6
TILE_LAYER       = "CartoDB positron"

COLOR_LOW        = "#ffffb2"   # kuning muda  → produksi rendah
COLOR_MID        = "#fd8d3c"   # oranye       → sedang
COLOR_HIGH       = "#bd0026"   # merah tua    → produksi tinggi


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: FORMAT ANGKA RIBUAN
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:,.0f}"

def _fmt2(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# RENDER PETA UTAMA
# ─────────────────────────────────────────────────────────────────────────────
def render_choropleth_map(
    merged_gdf: gpd.GeoDataFrame,
    year: int,
    mode: str = "Historis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> folium.Map:
    """
    Render peta choropleth Folium dari GeoDataFrame yang sudah di-merge.

    Parameters
    ----------
    merged_gdf : GeoDataFrame hasil merge_year_data()  —  8 baris, 1 per provinsi
    year       : int   —  tahun yang ditampilkan (untuk label)
    mode       : str   —  "Historis" atau "Proyeksi"
    vmin/vmax  : float —  batas warna (opsional; jika None dihitung dari data)

    Returns
    -------
    folium.Map siap di-render dengan streamlit-folium
    """
    gdf = merged_gdf.copy()

    # ── Tangani NaN pada kolom angka ────────────────────────────────────────
    num_cols = [TARGET_COL, "Curah hujan", "Kelembapan", "Suhu rata-rata", "Luas Panen"]
    for c in num_cols:
        if c not in gdf.columns:
            gdf[c] = np.nan

    # ── Skala warna ──────────────────────────────────────────────────────────
    prod_vals = gdf[TARGET_COL].dropna()
    _vmin = float(vmin) if vmin is not None else float(prod_vals.min()) if len(prod_vals) else 0
    _vmax = float(vmax) if vmax is not None else float(prod_vals.max()) if len(prod_vals) else 1
    if _vmax <= _vmin:
        _vmax = _vmin + 1

    colormap = cm.LinearColormap(
        colors=[COLOR_LOW, COLOR_MID, COLOR_HIGH],
        vmin=_vmin,
        vmax=_vmax,
        caption=f"Produksi Padi (ton) — {'Aktual' if mode == 'Historis' else 'Proyeksi'} {year}",
    )

    # ── Inisialisasi peta ────────────────────────────────────────────────────
    m = folium.Map(
        location=SUMATERA_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles=TILE_LAYER,
        prefer_canvas=True,
    )

    # ── GeoJson layer dengan style, tooltip, popup ───────────────────────────
    def _style(feature):
        produksi = feature["properties"].get(TARGET_COL)
        if produksi is None or (isinstance(produksi, float) and np.isnan(produksi)):
            color = "#cccccc"
        else:
            color = colormap(float(produksi))
        return {
            "fillColor"   : color,
            "color"       : "#444444",
            "weight"      : 1.2,
            "fillOpacity" : 0.80,
        }

    def _highlight(feature):
        return {
            "fillColor"   : "#ffffff",
            "color"       : "#222222",
            "weight"      : 2.5,
            "fillOpacity" : 0.50,
        }

    # Siapkan properties yang bisa dibaca GeoJSON widget
    # (GeoDataFrame harus dikonversi ke GeoJSON dict agar property terbawa)
    gdf_json = _prepare_geojson(gdf)

    tooltip = GeoJsonTooltip(
        fields=[PROV_COL, TARGET_COL],
        aliases=["🌾 Provinsi", "📦 Produksi (ton)"],
        localize=True,
        sticky=True,
        labels=True,
        style=(
            "background-color:#1e1e2e; color:#cdd6f4; "
            "font-family:monospace; font-size:13px; "
            "padding:8px 12px; border-radius:6px; border:none;"
        ),
    )

    popup = GeoJsonPopup(
        fields=[PROV_COL, TARGET_COL, "Suhu rata-rata", "Curah hujan", "Kelembapan", "Luas Panen"],
        aliases=[
            "Provinsi", "Produksi (ton)", "Suhu Rata-rata (°C)",
            "Curah Hujan (mm)", "Kelembapan (%)", "Luas Panen (ha)",
        ],
        localize=True,
        labels=True,
        max_width=280,
        style=(
            "background-color:#1e1e2e; color:#cdd6f4; "
            "font-family:sans-serif; font-size:13px;"
        ),
    )

    folium.GeoJson(
        gdf_json,
        name=f"Produksi Padi {year}",
        style_function=_style,
        highlight_function=_highlight,
        tooltip=tooltip,
        popup=popup,
    ).add_to(m)

    # ── Label nama provinsi di centroid ──────────────────────────────────────
    for _, row in gdf.iterrows():
        prov     = row.get(PROV_COL, "")
        produksi = row.get(TARGET_COL, np.nan)
        try:
            centroid = row.geometry.centroid
            cx, cy   = centroid.x, centroid.y
        except Exception:
            continue

        label_html = (
            f"<div style='"
            f"background:rgba(30,30,46,0.85);color:#cdd6f4;"
            f"padding:3px 7px;border-radius:4px;"
            f"font-size:10px;font-family:sans-serif;"
            f"text-align:center;line-height:1.4;"
            f"pointer-events:none;white-space:nowrap;'>"
            f"<b>{prov}</b><br>"
            f"{_fmt(produksi)} ton"
            f"</div>"
        )
        folium.Marker(
            location=[cy, cx],
            icon=folium.DivIcon(html=label_html, icon_size=(120, 40), icon_anchor=(60, 20)),
        ).add_to(m)

    # ── Colormap legend ──────────────────────────────────────────────────────
    colormap.add_to(m)

    # ── Layer control ─────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=True).add_to(m)

    return m


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: GeoDataFrame → GeoJSON dict dengan angka dibulatkan
# ─────────────────────────────────────────────────────────────────────────────
def _prepare_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """
    Konversi GeoDataFrame ke dict GeoJSON agar semua property
    (termasuk angka float) terbawa dengan benar ke folium.GeoJson.
    Angka NaN dikonversi ke None supaya JSON valid.
    """
    import json

    num_cols = [TARGET_COL, "Curah hujan", "Kelembapan", "Suhu rata-rata", "Luas Panen"]
    g = gdf.copy()
    for c in num_cols:
        if c in g.columns:
            g[c] = g[c].apply(lambda v: None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 2))

    # __geo_interface__ seringkali lebih cepat daripada .to_json() lalu json.loads
    return json.loads(g.to_json())
