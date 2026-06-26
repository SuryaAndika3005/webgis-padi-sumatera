# 🌾 Web GIS Prediksi Produksi Padi Sumatera

**Sistem Web GIS Interaktif untuk Visualisasi dan Prediksi Produksi Padi di Sumatera Berbasis Random Forest Regressor**

Aplikasi Web GIS spasio-temporal yang mengintegrasikan visualisasi data historis produksi padi (1993–2020) dengan proyeksi prediktif berbasis *machine learning* (2021–2025) untuk delapan provinsi di Pulau Sumatera, berdasarkan faktor-faktor iklim.

---

## 📖 Deskripsi Singkat

Produksi padi sangat bergantung pada kondisi iklim seperti curah hujan, kelembapan, dan suhu rata-rata. Aplikasi ini membantu memahami pola produksi padi secara spasial dan temporal, sekaligus mensimulasikan dampak skenario perubahan iklim terhadap produksi di masa depan melalui antarmuka peta interaktif.

Sistem menyediakan **dua mode eksplorasi**:

1. **Mode Historis** — menjelajahi data aktual produksi padi tahun 1993–2020 melalui *time slider*.
2. **Mode Proyeksi Cerdas** — mensimulasikan skenario iklim (mis. suhu +2%, curah hujan −15%) dan melihat prediksi produksi padi 2021–2025 secara langsung di peta.

---

## ✨ Fitur Utama

- 🗺️ **Peta Choropleth Interaktif** (Folium) dengan gradasi warna berdasarkan tingkat produksi
- 🖱️ **Tooltip & Popup** — *hover* menampilkan nama provinsi & produksi; *klik* menampilkan rincian suhu, curah hujan, dan kelembapan
- 📈 **Line Chart Interaktif** (Plotly) — tren historis (garis solid) tersambung dengan proyeksi (garis putus-putus)
- 🤖 **Model Random Forest Regressor** untuk prediksi produksi berbasis multivariat iklim
- 🎛️ **Slider Skenario Iklim** — modifikasi suhu, curah hujan, kelembapan, dan luas panen secara *real-time*
- 🧩 **Arsitektur Modular** — empat komponen terpisah yang mudah dipelihara

---

## 🖼️ Tangkapan Layar

| Mode Historis | Mode Proyeksi Cerdas |
|---|---|
| ![Mode Historis](docs/screenshot_historis.png) | ![Mode Proyeksi](docs/screenshot_proyeksi.png) |

> Letakkan tangkapan layar aplikasi pada folder `docs/` dan sesuaikan nama file di atas.

---

## 🏗️ Arsitektur Sistem

Aplikasi dirancang secara modular dalam empat komponen utama:

| Modul | File | Tanggung Jawab Utama |
|---|---|---|
| **Data Pipeline** | `data_pipeline.py` | Akuisisi data, pra-pemrosesan, *filtering* spasial, *caching* |
| **ML Engine** | `ml_engine.py` | Pelatihan Random Forest, proyeksi 5 tahun, pembangunan tren |
| **Map Visualizer** | `map_visualizer.py` | *Rendering* Folium choropleth, tooltip, popup interaktif |
| **App (UI)** | `app.py` | Antarmuka Streamlit, sidebar kontrol, *layout*, Plotly chart |

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ data_pipeline│ ──▶ │  ml_engine   │ ──▶ │  map_visualizer  │
│   (data)     │     │    (ML)      │     │      (peta)      │
└──────────────┘     └──────────────┘     └──────────────────┘
        │                    │                      │
        └────────────────────┴──────────────────────┘
                             ▼
                       ┌──────────┐
                       │  app.py  │  ◀── Antarmuka Streamlit
                       └──────────┘
```

---

## 📊 Dataset

| Dataset | File | Keterangan |
|---|---|---|
| Data Tanaman Padi Sumatera | `Data_Tanaman_Padi_Sumatera_version_1.csv` | Data historis 1993–2020, 8 provinsi (224 observasi). Target: `Produksi`. Fitur: `Luas Panen`, `Curah hujan`, `Kelembapan`, `Suhu rata-rata` |
| Batas Administrasi Provinsi | `38 Provinsi Indonesia - Provinsi.json` | GeoJSON batas administrasi 38 provinsi (WGS84 / EPSG:4326) |

**Delapan provinsi Sumatera** yang dianalisis: Aceh, Sumatera Utara, Sumatera Barat, Riau, Jambi, Sumatera Selatan, Bengkulu, dan Lampung.

---

## 🤖 Model Machine Learning

- **Algoritma:** Random Forest Regressor (`scikit-learn`)
- **Target:** Produksi padi (ton)
- **Fitur prediktor:** Luas Panen, Curah Hujan, Kelembapan, Suhu Rata-rata, dan Provinsi (*Label Encoding*)
- **Pra-pemrosesan:** penanganan *outlier* (IQR), *Label Encoding*, normalisasi `MinMaxScaler`
- **Pembagian data:** 80% latih / 20% uji

### Performa Model (data uji)

| Metrik | Nilai |
|---|---|
| R² (Koefisien Determinasi) | 0,91 |
| RMSE | 187.432 ton |
| MAE | 134.215 ton |

> ⚠️ Sesuaikan nilai metrik di atas dengan hasil aktual dari aplikasimu.

Dokumentasi lengkap pipeline ML tersedia di [`notebook/pipeline_ml.ipynb`](notebook/pipeline_ml.ipynb).

---

## ⚙️ Instalasi & Menjalankan

### 1. Klon repositori

```bash
git clone https://github.com/<username>/webgis-padi-sumatera.git
cd webgis-padi-sumatera
```

### 2. (Opsional) Buat *virtual environment*

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di `http://localhost:8501`.

> Jika perintah `streamlit` tidak dikenali, gunakan: `python -m streamlit run app.py`

---

## 📁 Struktur Proyek

```
webgis-padi-sumatera/
├── app.py                                  # Antarmuka utama Streamlit
├── data_pipeline.py                        # Modul data & spasial
├── ml_engine.py                            # Modul machine learning
├── map_visualizer.py                       # Modul peta Folium
├── requirements.txt                        # Daftar dependensi
├── README.md                               # Dokumentasi ini
├── Data_Tanaman_Padi_Sumatera_version_1.csv
├── 38 Provinsi Indonesia - Provinsi.json
├── docs/                                    # Tangkapan layar untuk dokumentasi
└── notebook/
    └── pipeline_ml.ipynb                    # Dokumentasi pipeline ML
```

---

## 📦 Dependensi Utama

`streamlit` · `streamlit-folium` · `folium` · `geopandas` · `pandas` · `numpy` · `scikit-learn` · `plotly` · `matplotlib`

---

## 🚀 Deployment

Aplikasi dapat di-*deploy* secara gratis melalui **Streamlit Community Cloud**:

1. *Push* seluruh kode ke repositori GitHub publik
2. Login ke [share.streamlit.io](https://share.streamlit.io)
3. Pilih repositori → atur *main file* ke `app.py` → klik **Deploy**

URL aplikasi: `https://<username>-webgis-padi-sumatera.streamlit.app`

---

## 👤 Penulis

**Surya Andika**
Program Studi Informatika, Fakultas Teknologi Informasi, Universitas Andalas

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik (Tugas Akhir Web GIS). Silakan sesuaikan lisensi sesuai kebutuhan, misalnya [MIT License](https://opensource.org/licenses/MIT).
