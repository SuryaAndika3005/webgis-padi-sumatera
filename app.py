"""
app.py — Web GIS Padi Sumatera (Leaflet.js embedded, smooth animation)
Jalankan: streamlit run app.py
"""

import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Web GIS Padi Sumatera",
    page_icon="🌾", layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp{background:#0f0f1a;color:#cdd6f4;font-family:'Inter',sans-serif}
.block-container{padding:1.2rem 1.8rem 2rem 1.8rem !important}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#181825 0%,#1e1e2e 100%);border-right:1px solid #313244}
[data-testid="stSidebar"] *{color:#cdd6f4 !important}
[data-testid="stSidebar"] hr{border-color:#313244 !important}
[data-testid="metric-container"]{background:#1e1e2e;border:1px solid #313244;border-radius:10px;padding:8px 12px !important}
[data-testid="metric-container"] label{font-size:.7rem !important;color:#a6adc8 !important}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:1rem !important;color:#cba6f7 !important}
h1{font-size:1.45rem !important;color:#cba6f7 !important;margin-bottom:.05rem !important}
h2,h3{color:#89b4fa !important;margin-top:.5rem !important}
h4{color:#a6e3a1 !important}
hr{border-color:#313244 !important;margin:.5rem 0 !important}
.stTabs [role="tablist"]{border-bottom:1px solid #313244}
.stTabs button[role="tab"]{color:#a6adc8 !important;font-size:.88rem;padding:6px 16px}
.stTabs button[aria-selected="true"]{color:#cba6f7 !important;border-bottom:2px solid #cba6f7 !important}
.stat-chip{display:inline-block;background:#313244;border-radius:8px;padding:5px 11px;font-size:.78rem;color:#cdd6f4;margin:3px 3px}
.stat-chip b{color:#cba6f7}
.prov-card{background:#1e1e2e;border:1px solid #313244;border-radius:12px;padding:14px 16px;margin-top:6px}
.placeholder-box{margin-top:36px;padding:52px 24px;text-align:center;background:#1e1e2e;border:1.5px dashed #45475a;border-radius:16px;color:#6c7086}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & TRAIN (cached)
# ─────────────────────────────────────────────────────────────────────────────
SUMATERA = ['Aceh','Sumatera Utara','Sumatera Barat','Riau',
            'Jambi','Sumatera Selatan','Bengkulu','Lampung']
FEATURES  = ['Luas Panen','Curah hujan','Kelembapan','Suhu rata-rata']
PALETTE   = ['#cba6f7','#89b4fa','#a6e3a1','#f38ba8',
             '#fab387','#f9e2af','#94e2d5','#74c7ec']

@st.cache_data(show_spinner="Memuat data …")
def load_data():
    return pd.read_csv("Data_Tanaman_Padi_Sumatera_version_1.csv")

@st.cache_resource(show_spinner="Melatih model …")
def train(_df):
    df_e = _df.copy()
    le   = LabelEncoder()
    df_e['penc'] = le.fit_transform(df_e['Provinsi'])
    mc   = FEATURES + ['penc','Tahun']
    X    = df_e[mc].values
    y    = df_e['Produksi'].values.reshape(-1,1)
    sx, sy = MinMaxScaler(), MinMaxScaler()
    Xs   = sx.fit_transform(X)
    ys   = sy.fit_transform(y).ravel()
    Xt,Xe,yt,ye = train_test_split(Xs,ys,test_size=.2,random_state=42)
    rf   = RandomForestRegressor(300,max_depth=10,min_samples_split=4,
                                  min_samples_leaf=2,n_jobs=-1,random_state=42)
    rf.fit(Xt,yt)
    yp   = sy.inverse_transform(rf.predict(Xe).reshape(-1,1)).ravel()
    yt_  = sy.inverse_transform(ye.reshape(-1,1)).ravel()
    mape = float(np.mean(np.abs((yt_-yp)/(yt_+1e-9)))*100)
    r2   = float(1 - np.sum((yt_-yp)**2)/np.sum((yt_-np.mean(yt_))**2))
    mae  = float(np.mean(np.abs(yt_-yp)))
    return rf, le, sx, sy, mc, {'R²':round(r2,4),'MAPE':round(mape,2),'MAE':round(mae,0)}

@st.cache_data(show_spinner="Menghitung proyeksi …")
def build_all_data(_df, _rf, _le, _sx, _sy, mc_tuple):
    mc = list(mc_tuple)
    BL = [2018,2019,2020]

    # Historical dict
    hist = {}
    for yr in sorted(_df['Tahun'].unique()):
        hist[int(yr)] = {}
        for _,row in _df[_df['Tahun']==yr].iterrows():
            hist[int(yr)][row['Provinsi']] = {
                'produksi': round(float(row['Produksi']),0),
                'luas_panen': round(float(row['Luas Panen']),0),
                'curah_hujan': round(float(row['Curah hujan']),1),
                'kelembapan': round(float(row['Kelembapan']),1),
                'suhu': round(float(row['Suhu rata-rata']),2),
            }

    # Projection dict per province
    proj = {}
    for prov in SUMATERA:
        bl  = _df[(_df['Tahun'].isin(BL))&(_df['Provinsi']==prov)][FEATURES].mean()
        enc = int(_le.transform([prov])[0])
        ts  = {int(yr): round(float(_df[(_df['Provinsi']==prov)&(_df['Tahun']==yr)]['Produksi'].values[0]),0)
               for yr in sorted(_df[_df['Provinsi']==prov]['Tahun'].unique())}
        clim= {}
        for yr in sorted(_df[_df['Provinsi']==prov]['Tahun'].unique()):
            r = _df[(_df['Provinsi']==prov)&(_df['Tahun']==yr)].iloc[0]
            clim[int(yr)]={'curah_hujan':round(float(r['Curah hujan']),1),
                           'kelembapan':round(float(r['Kelembapan']),1),
                           'suhu':round(float(r['Suhu rata-rata']),2),
                           'luas_panen':round(float(r['Luas Panen']),0)}
        for yr in range(2021,2026):
            fv  = [float(bl[c]) for c in FEATURES] + [enc, yr]
            Xr  = _sx.transform(np.array([fv]))
            p   = float(_sy.inverse_transform(_rf.predict(Xr).reshape(-1,1))[0,0])
            ts[yr] = round(max(p,0),0)
            clim[yr]={'curah_hujan':round(float(bl['Curah hujan']),1),
                      'kelembapan':round(float(bl['Kelembapan']),1),
                      'suhu':round(float(bl['Suhu rata-rata']),2),
                      'luas_panen':round(float(bl['Luas Panen']),0)}
        proj[prov] = {'timeseries':ts,'climate':clim}

    return hist, proj

df_hist = load_data()
rf, le, sx, sy, mc, metrics = train(df_hist)
hist_data, proj_data = build_all_data(df_hist, rf, le, sx, sy, tuple(mc))

# Load GeoJSON
with open("38 Provinsi Indonesia - Provinsi.json") as f:
    gj_raw = json.load(f)
feats = [ft for ft in gj_raw['features'] if ft['properties']['PROVINSI'] in SUMATERA]
for ft in feats:
    ft['properties']['Provinsi'] = ft['properties'].pop('PROVINSI')
    ft['properties'].pop('KODE_PROV', None)
sumatera_geojson = {'type':'FeatureCollection','features':feats}

PROD_MIN = float(df_hist['Produksi'].min())
PROD_MAX = float(df_hist['Produksi'].max())

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if 'chosen_prov' not in st.session_state:
    st.session_state.chosen_prov = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Web GIS Padi Sumatera")
    st.caption("Prediksi Produksi · Faktor Iklim")
    st.divider()

    st.markdown("### 🤖 Performa Model")
    st.caption("Random Forest · 80/20 split")
    c1,c2 = st.columns(2)
    c1.metric("R²",   f"{metrics['R²']:.4f}")
    c2.metric("MAPE", f"{metrics['MAPE']:.1f}%")
    st.metric("MAE",  f"{metrics['MAE']:,.0f} ton")

    st.divider()
    st.markdown("### 📌 Cara Pakai")
    st.markdown("""
- **Geser slider** atau **Play** untuk animasi peta
- **Klik provinsi** di peta untuk melihat detail
- Tab **Analisis** untuk proyeksi 2021–2025
""")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🌾 Produksi Padi Sumatera")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_map, tab_analisis = st.tabs(["🗺️  Peta Interaktif", "🔍  Analisis Prediksi"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PETA
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:

    # Embed semua data ke dalam HTML string
    geojson_str  = json.dumps(sumatera_geojson)
    histdata_str = json.dumps(hist_data)
    projdata_str = json.dumps(proj_data)

    map_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0f0f1a;font-family:'Inter',sans-serif;color:#cdd6f4;overflow:hidden}}
  #map{{width:100%;height:480px}}

  /* ── Controls bar ── */
  #controls{{
    display:flex;align-items:center;gap:10px;
    padding:8px 14px;background:#1e1e2e;
    border-top:1px solid #313244;flex-wrap:wrap;
  }}
  #year-display{{
    font-size:1.35rem;font-weight:800;color:#cba6f7;
    min-width:54px;text-align:center;letter-spacing:.03em;
  }}
  input[type=range]{{
    -webkit-appearance:none;flex:1;min-width:160px;max-width:420px;
    height:5px;border-radius:3px;
    background:linear-gradient(to right,#cba6f7 var(--pct,0%),#313244 var(--pct,0%));
    outline:none;cursor:pointer;
  }}
  input[type=range]::-webkit-slider-thumb{{
    -webkit-appearance:none;width:16px;height:16px;border-radius:50%;
    background:#cba6f7;border:2px solid #1e1e2e;cursor:pointer;
  }}
  .ctrl-btn{{
    padding:5px 16px;border-radius:8px;border:none;cursor:pointer;
    font-size:.82rem;font-weight:700;transition:all .15s;
  }}
  #btn-play{{background:#a6e3a1;color:#1e1e2e}}
  #btn-play:hover{{background:#94d3a2}}
  #btn-stop{{background:#f38ba8;color:#1e1e2e}}
  #btn-stop:hover{{background:#e07a95}}
  #btn-stop:disabled{{background:#45475a;color:#6c7086;cursor:not-allowed}}
  #speed-label{{font-size:.75rem;color:#a6adc8;white-space:nowrap}}
  #speed-sel{{background:#313244;color:#cdd6f4;border:1px solid #45475a;
              border-radius:6px;padding:3px 8px;font-size:.78rem;cursor:pointer}}

  /* ── Stats bar (bawah kontrol) ── */
  #statsbar{{
    display:flex;gap:10px;padding:6px 14px 8px;
    background:#181825;border-top:1px solid #313244;
    flex-wrap:wrap;align-items:center;
  }}
  .scard{{
    background:#1e1e2e;border:1px solid #313244;border-radius:8px;
    padding:5px 12px;font-size:.75rem;
  }}
  .scard span{{color:#a6adc8;font-size:.68rem;display:block}}
  .scard b{{color:#cba6f7;font-size:.88rem}}

  /* ── Popup custom (top-right) ── */
  #popup-panel{{
    display:none;
    position:absolute;top:10px;right:10px;z-index:900;
    background:#1e1e2e;border:1px solid #45475a;border-radius:12px;
    padding:14px 16px;min-width:200px;max-width:240px;
    box-shadow:0 4px 24px rgba(0,0,0,.6);
  }}
  #popup-panel h3{{
    font-size:.95rem;font-weight:700;color:#cba6f7;
    margin-bottom:8px;border-bottom:1px solid #313244;padding-bottom:6px;
  }}
  .pop-row{{display:flex;justify-content:space-between;
            font-size:.78rem;padding:3px 0;border-bottom:1px solid #23233a}}
  .pop-row:last-child{{border-bottom:none}}
  .pop-label{{color:#a6adc8}}
  .pop-val{{color:#cdd6f4;font-weight:600}}
  .pop-prod{{font-size:1.05rem;color:#a6e3a1;font-weight:800;
             text-align:center;margin:6px 0 4px;}}
  #popup-close{{
    position:absolute;top:8px;right:10px;
    background:none;border:none;color:#6c7086;
    cursor:pointer;font-size:1rem;line-height:1;
  }}
  #popup-close:hover{{color:#f38ba8}}

  /* ── Legend ── */
  #legend{{
    position:absolute;bottom:12px;left:10px;z-index:800;
    background:rgba(30,30,46,.92);border:1px solid #313244;
    border-radius:10px;padding:8px 12px;font-size:.72rem;min-width:140px;
  }}
  #legend b{{color:#cba6f7;font-size:.78rem}}
  .leg-bar{{
    height:10px;border-radius:4px;margin:5px 0 3px;
    background:linear-gradient(to right,#ffffb2,#fd8d3c,#bd0026);
  }}
  .leg-labels{{display:flex;justify-content:space-between;color:#a6adc8}}
</style>
</head>
<body>
<div style="position:relative">
  <div id="map"></div>
  <div id="popup-panel">
    <button id="popup-close" onclick="closePopup()">✕</button>
    <h3 id="pop-name"></h3>
    <div class="pop-prod" id="pop-prod"></div>
    <div class="pop-row"><span class="pop-label">Suhu</span><span class="pop-val" id="pop-suhu"></span></div>
    <div class="pop-row"><span class="pop-label">Curah Hujan</span><span class="pop-val" id="pop-hujan"></span></div>
    <div class="pop-row"><span class="pop-label">Kelembapan</span><span class="pop-val" id="pop-lembap"></span></div>
    <div class="pop-row"><span class="pop-label">Luas Panen</span><span class="pop-val" id="pop-lahan"></span></div>
  </div>
  <div id="legend">
    <b>Produksi (ton)</b>
    <div class="leg-bar"></div>
    <div class="leg-labels"><span id="leg-min"></span><span id="leg-max"></span></div>
  </div>
</div>

<div id="controls">
  <div id="year-display">1993</div>
  <input type="range" id="year-slider" min="1993" max="2020" value="1993" step="1"/>
  <button class="ctrl-btn" id="btn-play" onclick="startPlay()">▶ Play</button>
  <button class="ctrl-btn" id="btn-stop" onclick="stopPlay()" disabled>■ Stop</button>
  <span id="speed-label">Kecepatan:</span>
  <select id="speed-sel">
    <option value="1200">Lambat</option>
    <option value="700" selected>Normal</option>
    <option value="350">Cepat</option>
    <option value="150">Turbo</option>
  </select>
</div>

<div id="statsbar">
  <div class="scard"><span>Total Produksi</span><b id="s-total">—</b></div>
  <div class="scard"><span>Tertinggi</span><b id="s-high">—</b></div>
  <div class="scard"><span>Terendah</span><b id="s-low">—</b></div>
  <div class="scard"><span>Rerata Suhu</span><b id="s-suhu">—</b></div>
  <div class="scard"><span>Rerata Hujan</span><b id="s-hujan">—</b></div>
</div>

<script>
// ── Embedded data ────────────────────────────────────────────────────────────
const GEOJSON   = {geojson_str};
const HIST_DATA = {histdata_str};
const PROD_MIN  = {PROD_MIN};
const PROD_MAX  = {PROD_MAX};

// ── Leaflet init ─────────────────────────────────────────────────────────────
const map = L.map('map', {{
  center: [-0.8, 102.5], zoom: 6,
  zoomControl: true, attributionControl: false,
}});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom: 12
}}).addTo(map);

// ── Color scale ──────────────────────────────────────────────────────────────
function lerp(a,b,t){{ return a+(b-a)*t; }}
function hexToRgb(hex){{
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  return [r,g,b];
}}
function rgbToHex(r,g,b){{
  return '#'+[r,g,b].map(v=>Math.round(v).toString(16).padStart(2,'0')).join('');
}}
const C_LOW=[255,255,178], C_MID=[253,141,60], C_HIGH=[189,0,38];

function prodColor(val){{
  if(val===null||val===undefined) return '#45475a';
  const t = Math.max(0,Math.min(1,(val-PROD_MIN)/(PROD_MAX-PROD_MIN)));
  let r,g,b;
  if(t<0.5){{
    const tt=t*2;
    r=lerp(C_LOW[0],C_MID[0],tt);g=lerp(C_LOW[1],C_MID[1],tt);b=lerp(C_LOW[2],C_MID[2],tt);
  }}else{{
    const tt=(t-0.5)*2;
    r=lerp(C_MID[0],C_HIGH[0],tt);g=lerp(C_MID[1],C_HIGH[1],tt);b=lerp(C_MID[2],C_HIGH[2],tt);
  }}
  return rgbToHex(r,g,b);
}}

// Legend
document.getElementById('leg-min').textContent = (PROD_MIN/1e6).toFixed(1)+' Jt';
document.getElementById('leg-max').textContent = (PROD_MAX/1e6).toFixed(1)+' Jt';

// ── GeoJSON layer ─────────────────────────────────────────────────────────────
let currentYear = 1993;
let geojsonLayer;
let selectedLayer = null;

function styleFeature(feat){{
  const prov = feat.properties.Provinsi;
  const d    = HIST_DATA[currentYear]?.[prov];
  return {{
    fillColor  : prodColor(d?.produksi),
    fillOpacity: 0.82,
    color      : '#2a2a3e',
    weight     : 1.5,
  }};
}}

function highlightFeature(e){{
  const layer = e.target;
  if(layer !== selectedLayer){{
    layer.setStyle({{color:'#cba6f7',weight:2.5,fillOpacity:0.9}});
  }}
}}
function resetHighlight(e){{
  const layer = e.target;
  if(layer !== selectedLayer){{
    geojsonLayer.resetStyle(layer);
  }}
}}
function onFeatureClick(e){{
  const prov = e.target.feature.properties.Provinsi;
  showPopup(prov, currentYear);
  // Highlight selected
  if(selectedLayer){{ geojsonLayer.resetStyle(selectedLayer); }}
  selectedLayer = e.target;
  selectedLayer.setStyle({{color:'#f5c2e7',weight:3,fillOpacity:0.95}});
}}

function onEachFeature(feat, layer){{
  layer.on({{
    mouseover: highlightFeature,
    mouseout : resetHighlight,
    click    : onFeatureClick,
  }});
}}

geojsonLayer = L.geoJSON(GEOJSON, {{
  style      : styleFeature,
  onEachFeature: onEachFeature,
}}).addTo(map);
map.fitBounds(geojsonLayer.getBounds(), {{padding:[10,10]}});

// ── Popup panel ───────────────────────────────────────────────────────────────
function showPopup(prov, yr){{
  const d = HIST_DATA[yr]?.[prov];
  if(!d) return;
  document.getElementById('pop-name').textContent  = prov;
  document.getElementById('pop-prod').textContent  = (d.produksi/1e6).toFixed(2)+' Jt ton';
  document.getElementById('pop-suhu').textContent  = d.suhu+' °C';
  document.getElementById('pop-hujan').textContent = d.curah_hujan.toLocaleString()+' mm';
  document.getElementById('pop-lembap').textContent= d.kelembapan+' %';
  document.getElementById('pop-lahan').textContent = d.luas_panen.toLocaleString()+' ha';
  document.getElementById('popup-panel').style.display='block';
}}
function closePopup(){{
  document.getElementById('popup-panel').style.display='none';
  if(selectedLayer){{ geojsonLayer.resetStyle(selectedLayer); selectedLayer=null; }}
}}

// ── Stats bar ─────────────────────────────────────────────────────────────────
function updateStats(yr){{
  const yd = HIST_DATA[yr]; if(!yd) return;
  const vals = Object.values(yd);
  const total = vals.reduce((a,b)=>a+b.produksi,0);
  const high  = vals.reduce((a,b)=>b.produksi>a.produksi?b:a);
  const low   = vals.reduce((a,b)=>b.produksi<a.produksi?b:a);
  const avgS  = vals.reduce((a,b)=>a+b.suhu,0)/vals.length;
  const avgH  = vals.reduce((a,b)=>a+b.curah_hujan,0)/vals.length;
  document.getElementById('s-total').textContent = (total/1e6).toFixed(2)+' Jt ton';
  document.getElementById('s-high').textContent  = high.produksi.toLocaleString('id')+' ton';
  document.getElementById('s-low').textContent   = low.produksi.toLocaleString('id')+' ton';
  document.getElementById('s-suhu').textContent  = avgS.toFixed(1)+' °C';
  document.getElementById('s-hujan').textContent = avgH.toFixed(0)+' mm';
}}

// ── Update map ────────────────────────────────────────────────────────────────
function updateMap(yr){{
  currentYear = yr;
  document.getElementById('year-display').textContent = yr;
  const slider = document.getElementById('year-slider');
  slider.value = yr;
  const pct = ((yr-1993)/(2020-1993)*100).toFixed(1);
  slider.style.setProperty('--pct', pct+'%');
  // Smooth color transition — just update style, no layer recreate
  geojsonLayer.setStyle(styleFeature);
  updateStats(yr);
  // Update popup if open
  const pp = document.getElementById('popup-panel');
  if(pp.style.display==='block'){{
    const name = document.getElementById('pop-name').textContent;
    showPopup(name, yr);
  }}
}}
updateMap(1993);

// ── Slider input ──────────────────────────────────────────────────────────────
document.getElementById('year-slider').addEventListener('input', function(){{
  stopPlay();
  updateMap(parseInt(this.value));
}});

// ── Animation ─────────────────────────────────────────────────────────────────
let playTimer = null;
function startPlay(){{
  if(playTimer) return;
  document.getElementById('btn-play').disabled = true;
  document.getElementById('btn-stop').disabled = false;
  let yr = parseInt(document.getElementById('year-slider').value);
  if(yr >= 2020) yr = 1993;

  function tick(){{
    updateMap(yr);
    if(yr < 2020){{
      yr++;
      const spd = parseInt(document.getElementById('speed-sel').value);
      playTimer = setTimeout(tick, spd);
    }} else {{
      stopPlay();
    }}
  }}
  const spd = parseInt(document.getElementById('speed-sel').value);
  playTimer = setTimeout(tick, spd);
}}
function stopPlay(){{
  if(playTimer){{ clearTimeout(playTimer); playTimer=null; }}
  document.getElementById('btn-play').disabled  = false;
  document.getElementById('btn-stop').disabled  = true;
}}

// Map click on empty area closes popup
map.on('click', function(e){{
  if(!e.originalEvent.target.closest('.leaflet-interactive')){{
    closePopup();
  }}
}});
</script>
</body>
</html>"""

    components.html(map_html, height=600, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALISIS PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_analisis:
    st.markdown("### 🔍 Analisis & Prediksi per Provinsi")
    st.markdown(
        "Pilih provinsi untuk melihat **tren historis**, "
        "**proyeksi variabel iklim** (via regresi linear), "
        "dan **prediksi produksi 2021–2025** dari model Random Forest."
    )
    st.markdown("")

    sel_col, _, __ = st.columns([2, 1, 1])
    with sel_col:
        chosen = st.selectbox(
            "🌾 Pilih Provinsi",
            ["— Pilih provinsi —"] + SUMATERA,
            index=0, key="prov_select",
        )

    if chosen == "— Pilih provinsi —":
        st.markdown("""
        <div class="placeholder-box">
          <div style="font-size:2.8rem;margin-bottom:10px">🌾</div>
          <div style="font-size:1rem;font-weight:600;color:#a6adc8">
            Pilih provinsi di atas untuk memulai analisis
          </div>
          <div style="font-size:.82rem;margin-top:6px">
            Sistem menampilkan tren historis 1993–2020, proyeksi variabel iklim,
            dan prediksi produksi 2021–2025
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        prov = chosen
        # Hitung proyeksi on-demand (pakai ml_engine yang sudah diimport)
        from ml_engine import project_province, build_combined_timeseries
        df_prov_proj, trend_meta = project_province(
            df_hist, prov, rf, le, sx, sy, mc
        )
        df_prov_hist = df_hist[df_hist["Provinsi"] == prov].sort_values("Tahun")
        hist_yr  = sorted(df_prov_hist["Tahun"].tolist())
        proj_yr  = sorted(df_prov_proj["Tahun"].tolist())

        last_prod  = float(df_prov_hist[df_prov_hist["Tahun"] == 2020]["Produksi"].values[0])
        pred_2025  = float(df_prov_proj[df_prov_proj["Tahun"] == 2025]["Produksi"].values[0])
        max_yr     = int(df_prov_hist.loc[df_prov_hist["Produksi"].idxmax(), "Tahun"])
        min_yr     = int(df_prov_hist.loc[df_prov_hist["Produksi"].idxmin(), "Tahun"])
        max_prod   = float(df_prov_hist["Produksi"].max())
        min_prod   = float(df_prov_hist["Produksi"].min())
        growth_pct = (pred_2025 - last_prod) / (last_prod + 1e-9) * 100
        g_col      = "#a6e3a1" if growth_pct >= 0 else "#f38ba8"
        g_arrow    = "▲" if growth_pct >= 0 else "▼"

        # ── Kartu ringkas ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="prov-card">
          <div style="font-size:1.1rem;font-weight:700;color:#cba6f7;margin-bottom:10px">📍 {prov}</div>
          <div>
            <span class="stat-chip">Produksi 2020: <b>{last_prod:,.0f} ton</b></span>
            <span class="stat-chip">Tertinggi: <b>{max_prod:,.0f} ton</b> ({max_yr})</span>
            <span class="stat-chip">Terendah: <b>{min_prod:,.0f} ton</b> ({min_yr})</span>
            <span class="stat-chip" style="color:{g_col}">
              Proyeksi 2025: <b>{pred_2025:,.0f} ton</b> &nbsp;
              {g_arrow} {abs(growth_pct):.1f}% vs 2020
            </span>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("")

        # ── Chart 1: Produksi historis + proyeksi ─────────────────────────────
        st.markdown("#### 📈 Tren Produksi & Proyeksi")
        prov_idx = SUMATERA.index(prov)
        fig_prod = go.Figure()

        hist_y = df_prov_hist["Produksi"].tolist()
        proj_y = df_prov_proj["Produksi"].tolist()

        fig_prod.add_trace(go.Scatter(
            x=hist_yr, y=hist_y,
            mode="lines+markers", name="Historis",
            line=dict(color="#89b4fa", width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} ton<extra>Historis</extra>",
        ))
        conn_x = [hist_yr[-1], proj_yr[0]]
        conn_y = [hist_y[-1], proj_y[0]]
        fig_prod.add_trace(go.Scatter(
            x=conn_x, y=conn_y, mode="lines", showlegend=False,
            line=dict(color="#cba6f7", width=1.5, dash="dot"), hoverinfo="skip",
        ))
        fig_prod.add_trace(go.Scatter(
            x=proj_yr, y=proj_y,
            mode="lines+markers", name="Proyeksi RF (iklim tren)",
            line=dict(color="#cba6f7", width=2.5, dash="dash"),
            marker=dict(size=10, symbol="diamond", color="#cba6f7",
                        line=dict(color="#1e1e2e", width=2)),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} ton<extra>Proyeksi</extra>",
        ))
        for i, yr in enumerate(proj_yr):
            fig_prod.add_annotation(
                x=yr, y=proj_y[i],
                text=f"<b>{proj_y[i]/1e6:.2f} Jt</b>",
                showarrow=False, yshift=17, font=dict(color="#cba6f7", size=10),
            )
        fig_prod.add_vrect(x0=2020.5, x1=2025.5, fillcolor="#313244",
                           opacity=0.28, layer="below", line_width=0)
        fig_prod.add_vline(x=2020.5, line_dash="dash", line_color="#45475a",
                           line_width=1.2, annotation_text="  Batas Proyeksi",
                           annotation_font_color="#6c7086", annotation_position="top left")
        fig_prod.update_layout(
            paper_bgcolor="#0f0f1a", plot_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4", size=12), height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.06, x=0,
                        bgcolor="rgba(30,30,46,.85)", bordercolor="#313244", borderwidth=1),
            xaxis=dict(title="Tahun", gridcolor="#313244", tickmode="linear",
                       dtick=3, range=[1992, 2026]),
            yaxis=dict(title="Produksi (ton)", gridcolor="#313244", tickformat=",.0f"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_prod, use_container_width=True)

        # ── Chart 2: Proyeksi variabel iklim (4 sub-chart) ────────────────────
        st.markdown("#### 🌡️ Proyeksi Variabel Input Model (Berdasarkan Tren Historis)")
        st.caption(
            "Setiap variabel diproyeksikan menggunakan **regresi linear OLS** atas data historis 1993–2020. "
            "Tanda 🔴 = tren turun, 🟢 = tren naik — nilai ini menjadi input nyata ke model RF tiap tahunnya."
        )

        feat_cfg = {
            "Luas Panen"     : {"unit": "ha",  "color": "#a6e3a1", "label": "Luas Panen (ha)"},
            "Curah hujan"    : {"unit": "mm",  "color": "#89b4fa", "label": "Curah Hujan (mm)"},
            "Kelembapan"     : {"unit": "%",   "color": "#94e2d5", "label": "Kelembapan (%)"},
            "Suhu rata-rata" : {"unit": "°C",  "color": "#f38ba8", "label": "Suhu Rata-rata (°C)"},
        }

        col_pairs = list(feat_cfg.items())
        row1 = st.columns(2)
        row2 = st.columns(2)
        col_list = [row1[0], row1[1], row2[0], row2[1]]

        for col_widget, (feat, cfg) in zip(col_list, col_pairs):
            with col_widget:
                meta  = trend_meta[feat]
                slope = meta["slope"]
                r2    = meta["r2"]
                trend_icon = "🟢" if slope >= 0 else "🔴"
                trend_txt  = f"+{slope:.3f}" if slope >= 0 else f"{slope:.3f}"

                # Data historis
                h_x = df_prov_hist["Tahun"].tolist()
                h_y = df_prov_hist[feat].tolist()

                # Data proyeksi
                p_x = proj_yr
                p_y = [float(df_prov_proj[df_prov_proj["Tahun"] == yr][feat].values[0])
                       for yr in proj_yr]

                # Garis tren OLS full-period (extend ke 2025)
                from scipy import stats as sp
                sl, ic, _, _, _ = sp.linregress(h_x, h_y)
                trend_x = list(range(1993, 2026))
                trend_y = [ic + sl * yr for yr in trend_x]

                fig_feat = go.Figure()
                # Tren line (background)
                fig_feat.add_trace(go.Scatter(
                    x=trend_x, y=trend_y, mode="lines",
                    name="Tren OLS",
                    line=dict(color=cfg["color"], width=1, dash="dot"),
                    opacity=0.4, hoverinfo="skip",
                ))
                # Historis scatter
                fig_feat.add_trace(go.Scatter(
                    x=h_x, y=h_y, mode="lines+markers",
                    name="Historis",
                    line=dict(color=cfg["color"], width=1.8),
                    marker=dict(size=4, color=cfg["color"]),
                    hovertemplate=f"%{{x}}: %{{y:.2f}} {cfg['unit']}<extra>Historis</extra>",
                ))
                # Proyeksi markers
                fig_feat.add_trace(go.Scatter(
                    x=p_x, y=p_y, mode="markers+lines",
                    name="Proyeksi",
                    line=dict(color=cfg["color"], width=2, dash="dash"),
                    marker=dict(size=9, symbol="diamond", color=cfg["color"],
                                line=dict(color="#1e1e2e", width=1.5)),
                    hovertemplate=f"%{{x}}: %{{y:.2f}} {cfg['unit']}<extra>Proyeksi</extra>",
                ))
                fig_feat.add_vrect(x0=2020.5, x1=2025.5, fillcolor="#313244",
                                   opacity=0.25, layer="below", line_width=0)
                fig_feat.update_layout(
                    title=dict(
                        text=f"{trend_icon} {cfg['label']}<br>"
                             f"<span style='font-size:10px;color:#a6adc8'>"
                             f"slope={trend_txt}/thn · R²={r2:.3f}</span>",
                        font=dict(size=12, color="#cdd6f4"), x=0,
                    ),
                    paper_bgcolor="#0f0f1a", plot_bgcolor="#1e1e2e",
                    font=dict(color="#cdd6f4", size=10),
                    height=240, margin=dict(l=6, r=6, t=58, b=6),
                    showlegend=False,
                    xaxis=dict(gridcolor="#313244", tickmode="linear",
                               dtick=5, range=[1992, 2026]),
                    yaxis=dict(gridcolor="#313244"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_feat, use_container_width=True)

        # ── Tabel rincian proyeksi ─────────────────────────────────────────────
        st.markdown("#### 📋 Rincian Lengkap Proyeksi 2021–2025")
        st.caption("Nilai variabel iklim di bawah adalah hasil proyeksi regresi linear, bukan baseline statis.")

        rows = []
        for _, row in df_prov_proj.iterrows():
            rows.append({
                "Tahun"                  : int(row["Tahun"]),
                "Prediksi Produksi (ton)": f"{row['Produksi']:,.0f}",
                "Luas Panen (ha)"        : f"{row['Luas Panen']:,.0f}",
                "Suhu (°C)"              : f"{row['Suhu rata-rata']:.2f}",
                "Curah Hujan (mm)"       : f"{row['Curah hujan']:.1f}",
                "Kelembapan (%)"         : f"{row['Kelembapan']:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=215)

        # ── Tabel tren variabel ────────────────────────────────────────────────
        with st.expander("📊 Detail Tren Regresi per Variabel"):
            st.caption("Slope = perubahan per tahun. R² mendekati 1 = tren sangat konsisten.")
            trend_rows = []
            for feat, meta in trend_meta.items():
                slope = meta["slope"]
                arah  = "Naik ↑" if slope > 0 else "Turun ↓"
                signif = "✅ Kuat" if meta["r2"] > 0.3 else ("⚠️ Lemah" if meta["r2"] > 0.05 else "❌ Sangat lemah")
                trend_rows.append({
                    "Variabel"   : feat,
                    "Slope/thn"  : f"{slope:+.4f}",
                    "Arah Tren"  : arah,
                    "R²"         : f"{meta['r2']:.4f}",
                    "Kekuatan"   : signif,
                    "p-value"    : f"{meta['p_val']:.4f}",
                })
            st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

        # ── Expander: historis lengkap ─────────────────────────────────────────
        with st.expander("📂 Data Historis Lengkap (1993–2020)"):
            hdisp = df_prov_hist[["Tahun","Produksi","Luas Panen",
                                   "Curah hujan","Kelembapan","Suhu rata-rata"]].copy()
            for c in ["Produksi","Luas Panen"]:
                hdisp[c] = hdisp[c].apply(lambda v: f"{v:,.0f}")
            hdisp = hdisp.rename(columns={
                "Produksi"      : "Produksi (ton)",
                "Luas Panen"    : "Luas Panen (ha)",
                "Curah hujan"   : "Curah Hujan (mm)",
                "Kelembapan"    : "Kelembapan (%)",
                "Suhu rata-rata": "Suhu (°C)",
            })
            st.dataframe(hdisp, use_container_width=True, hide_index=True, height=300)


# FOOTER
st.divider()
st.caption("🌾 Web GIS Prediksi Produksi Padi Regional Sumatera · Random Forest · Data BPS/BMKG 1993–2020 · Tugas Akhir")
