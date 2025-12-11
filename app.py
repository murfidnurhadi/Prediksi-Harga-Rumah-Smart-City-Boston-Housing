# ============================================
# SMART CITY – Boston Housing Analytics Dashboard
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import folium
import plotly.express as px
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Smart City • Boston Housing",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ Smart City Dashboard – Boston Housing Analytics")
st.markdown("""
Sistem analitik untuk mendukung **perencanaan kota cerdas (smart city)**  
berbasis prediksi harga rumah dan analisis faktor-faktor urban.
""")

# ==========================================================
# 1. DATA LOADER (LOCAL FILE / GITHUB RAW / GOOGLE DRIVE / UPLOAD)
# ==========================================================

import pandas as pd
import streamlit as st


# ===============================
# 📌 LOAD DATA DARI FILE LOKAL
# ===============================
@st.cache_data
def load_from_local():
    try:
        df = pd.read_csv("BostonHousing.csv")
        return df
    except:
        return None


# ===============================
# 📌 LOAD DATA DARI GITHUB RAW
# ===============================
@st.cache_data
def load_from_github(url):
    try:
        df = pd.read_csv(url)
        return df
    except:
        return None


# ===============================
# 📌 LOAD DATA DARI GOOGLE DRIVE
# ===============================
@st.cache_data
def load_from_gdrive(share_url):
    try:
        # Ambil file ID dari link Google Drive
        file_id = share_url.split("/d/")[1].split("/")[0]

        # Mengubah ke direct download URL
        dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        # Load CSV
        df = pd.read_csv(dl_url)

        # Bersihkan nama kolom
        df.columns = df.columns.str.lower().str.strip()

        return df

    except Exception as e:
        st.error(f"❌ Gagal memproses link Google Drive: {e}")
        return None


# ===============================
# 📌 LOAD DATA DARI FILE UPLOAD
# ===============================
@st.cache_data
def load_from_upload(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except:
        return None


# ===============================
# 📌 SIDEBAR MENU LOAD DATA
# ===============================
st.sidebar.header("📂 Load Dataset Boston Housing")

source = st.sidebar.selectbox(
    "Pilih sumber data:",
    ["Lokal (folder yang sama)", "GitHub Raw URL", "Google Drive", "Upload Manual"]
)

df = None


# ===============================
# 📌 PILIHAN 1: LOAD LOKAL
# ===============================
if source == "Lokal (folder yang sama)":
    df = load_from_local()
    if df is not None:
        st.sidebar.success("📁 Berhasil load file lokal.")
    else:
        st.sidebar.error("❌ File lokal 'BostonHousing.csv' tidak ditemukan.")


# ===============================
# 📌 PILIHAN 2: LOAD GITHUB RAW
# ===============================
elif source == "GitHub Raw URL":

    st.sidebar.info("Dataset otomatis di-load dari folder /dataset di GitHub.")

    github_raw_url = "https://raw.githubusercontent.com/murfidnurhadi/Prediksi-Harga-Rumah-Smart-City-Boston-Housing/main/dataset/BostonHousing.csv"

    df = load_from_github(github_raw_url)

    if df is not None:
        st.sidebar.success("📡 Berhasil load dataset dari GitHub!")
    else:
        st.sidebar.error("❌ Dataset tidak ditemukan di GitHub. Periksa nama file & repositori.")

# ===============================
# 📌 PILIHAN 3: LOAD GOOGLE DRIVE
# ===============================
elif source == "Google Drive":

    # ==== Link Google Drive Bawaan Anda ====
    default_gdrive_link = "https://drive.google.com/file/d/1GXfcCKjJBGGmCGalJrIsCMKnTe4qujT7/view?usp=sharing"

    url = st.sidebar.text_input(
        "Paste link Google Drive CSV (share link):",
        value=default_gdrive_link  # <-- otomatis terisi, tidak perlu input
    )

    # ==== PROSES LANGSUNG TANPA HARUS KLIK APA-APA ====
    if url == default_gdrive_link:
        df = load_from_gdrive(default_gdrive_link)
        if df is not None:
            st.sidebar.success("🔗 Dataset Google Drive berhasil dimuat otomatis!")
        else:
            st.sidebar.error("❌ Gagal memuat dataset default Google Drive.")
    else:
        # Jika user mengganti link manual
        df = load_from_gdrive(url)
        if df is not None:
            st.sidebar.success("🔗 Berhasil load dari Google Drive!")
        else:
            st.sidebar.error("❌ Link Google Drive tidak valid.")

# ===============================
# 📌 PILIHAN 4: UPLOAD MANUAL
# ===============================
elif source == "Upload Manual":
    uploaded = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
    if uploaded:
        df = load_from_upload(uploaded)
        if df is not None:
            st.sidebar.success("📤 Upload berhasil!")
        else:
            st.sidebar.error("❌ File tidak dapat dibaca.")


# ===============================
# 📌 VALIDASI TERAKHIR
# ===============================
if df is None:
    st.error("❌ Dataset belum berhasil dimuat. Silakan pilih sumber data.")
    st.stop()

# Bersihkan dataset
df.columns = df.columns.str.lower().str.strip()
df.dropna(inplace=True)

st.success("✅ Dataset berhasil dimuat dan siap digunakan!")


# ==========================================================
# 2. SIMULASI KOORDINAT (UNTUK PEMETAAN SMART CITY)
# ==========================================================

@st.cache_data
def generate_coordinates(df):
    np.random.seed(42)
    df["lat"] = 42.2 + np.random.normal(0, 0.02, size=len(df))   # Boston area latitude
    df["lon"] = -71.0 + np.random.normal(0, 0.03, size=len(df))  # Boston area longitude
    return df

df = generate_coordinates(df)

# ==========================================================
# 3. NAVIGATION TABS
# ==========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Korelasi & Insight", 
    "📍 Peta Smart City", 
    "📈 Regresi Linear", 
    "🔢 Prediksi Manual", 
    "📑 Statistik"
])

# ==========================================================
# 📊 TAB 1 — Korelasi
# ==========================================================
with tab1:
    st.subheader("📊 Analisis Korelasi Fitur Urban")

    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        labels=dict(color="Korelasi"),
    )
    fig.update_traces(hovertemplate="Fitur 1: %{y}<br>Fitur 2: %{x}<br>Korelasi: %{z:.3f}")
    st.plotly_chart(fig, use_container_width=True)
# ==========================================================
# 📍 TAB 2 — PETA
# ==========================================================
with tab2:

    st.subheader("📍 Peta Distribusi Harga Rumah Boston")

    # Layout 2 kolom: peta kiri, info kanan
    col_map, col_info = st.columns([2, 1])

    # ========== Panel Info Kanan ==========
    with col_info:
        info_box = st.empty()
        info_box.info("Klik titik pada peta untuk melihat informasi rumah.")

    # ========== BANGUN PETA ==========
    with col_map:
        m = folium.Map(location=[42.32, -71.05], zoom_start=12)

        # Tambahkan marker
        for idx, row in df.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5,
                color="blue",
                fill=True,
                fill_opacity=0.7,
                tooltip=f"MEDV {row['medv']} | RM {row['rm']} | LSTAT {row['lstat']}%"
            ).add_to(m)

        # Gunakan streamlit-folium CLICK EVENT RESMI
        map_data = st_folium(
            m,
            width=900,
            height=550,
            key="smartcity_map",
            returned_objects=["last_clicked"]
        )

    # ========== PROSES KLIKAN ==========
    if map_data and map_data.get("last_clicked"):
        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]

        # cari marker terdekat
        df["dist"] = ((df["lat"] - click_lat)**2 + (df["lon"] - click_lon)**2)**0.5
        row = df.loc[df["dist"].idxmin()]  # paling dekat

        # tampilkan info di panel
        with col_info:
            info_box.markdown(f"""
            ### 📝 Informasi Rumah Terpilih

            **🏠 Harga Rumah (MEDV)**  
            `{row['medv']} ribu dolar`  
            *MEDV = Median value (dalam ribu dolar)*

            **🛏 RM — Average Rooms**  
            `{row['rm']} kamar`  
            *Rata-rata jumlah kamar per rumah*

            **📉 LSTAT — % Low Status Population**  
            `{row['lstat']}% penduduk berstatus ekonomi rendah`

            **📌 Koordinat**  
            Lat: `{row['lat']}`  
            Lon: `{row['lon']}`  
            """)

# ==========================================================
# 📈 TAB 3 — REGRESI LINEAR
# ==========================================================
with tab3:

    st.subheader("📈 Model Regresi Linear – Prediksi Harga Rumah")

    # ============================
    # PILIH FITUR REGRESI
    # ============================
    fitur = st.multiselect(
        "Pilih fitur independen:",
        options=list(df.columns.drop(["medv", "lat", "lon"])),
        default=["rm", "lstat", "ptratio"]
    )

    if len(fitur) == 0:
        st.warning("Pilih minimal satu fitur regresi.")
        st.stop()

    X = df[fitur]
    y = df["medv"]

    # ============================
    # TRAIN TEST SPLIT
    # ============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ============================
    # TRAIN MODEL
    # ============================
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ============================
    # METRIK EVALUASI
    # ============================
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("RMSE", f"{rmse:.3f}")

    # ============================
    # GRAFIK INTERAKTIF PREDIKSI vs AKTUAL
    # ============================
    st.write("### 📉 Prediksi vs Aktual (Interaktif)")

    plot_df = pd.DataFrame({
        "Aktual": y_test,
        "Prediksi": y_pred,
        "Error": y_pred - y_test
    })

    fig = px.scatter(
        plot_df,
        x="Aktual",
        y="Prediksi",
        color="Error",
        color_continuous_scale="RdBu",
        opacity=0.8,
        title="Perbandingan Nilai Aktual vs Prediksi",
        labels={
            "Aktual": "Nilai MEDV Asli",
            "Prediksi": "Nilai Prediksi",
            "Error": "Selisih (Prediksi - Aktual)"
        }
    )

    fig.add_shape(
        type="line",
        x0=plot_df["Aktual"].min(),
        y0=plot_df["Aktual"].min(),
        x1=plot_df["Aktual"].max(),
        y1=plot_df["Aktual"].max(),
        line=dict(color="green", width=2, dash="dash")
    )

    fig.update_traces(
        hovertemplate="<b>Aktual:</b> %{x:.2f}<br>"
                      "<b>Prediksi:</b> %{y:.2f}<br>"
                      "<b>Error:</b> %{marker.color:.2f}"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================
    # KOEFISIEN REGRESI
    # ============================
    st.write("### 🔢 Koefisien Model Regresi")
    coef_df = pd.DataFrame({"Fitur": fitur, "Koefisien": model.coef_})
    st.dataframe(coef_df.style.format({"Koefisien": "{:.5f}"}))
# ==========================================================
# 🔢 TAB 4 — PREDIKSI MANUAL
# ==========================================================
with tab4:
    st.subheader("🔢 Prediksi Harga Rumah – Input Manual")

    user_input = {}
    for f in fitur:
        user_input[f] = st.number_input(
            f"Masukkan nilai {f.upper()}",
            value=float(df[f].mean())
        )

    if st.button("Prediksi Harga"):
        pred = model.predict(pd.DataFrame([user_input]))[0]
        st.success(f"💰 Harga rumah diprediksi: **{pred:.2f} ribu dolar**")

# ==========================================================
# 📑 TAB 5 — STATISTIK
# ==========================================================
with tab5:
    st.subheader("📑 Statistik Deskriptif")
    st.dataframe(df.describe().T)
