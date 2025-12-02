
import streamlit as st
import pandas as pd
import numpy as np
import skicit-learn as sklearn
import pickle

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Tingkat Stres Mahasiswa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Pemetaan Kategori Hasil ---
# Model memprediksi skor kontinu. Kita memetakan skor tersebut ke kategori 0, 1, atau 2.
def map_prediction_to_level(score):
    # Karena stress_level di dataset asli adalah 0, 1, dan 2,
    # kita bisa menggunakan batas sederhana atau membulatkannya.
    # Batas yang masuk akal adalah 0.5 dan 1.5.
    if score < 0.5:
        return "Tingkat Stres Rendah (0)"
    elif score < 1.5:
        return "Tingkat Stres Sedang (1)"
    else:
        return "Tingkat Stres Tinggi (2)"

# --- Memuat Model ---
@st.cache_resource
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

MODEL_PATH = "linear_regression_model.pkl"
model = load_model(MODEL_PATH)

# --- Judul dan Deskripsi Aplikasi ---
st.title("🧠 Prediktor Tingkat Stres Mahasiswa")

if model is not None:
    st.sidebar.header("Input Fitur Mahasiswa")
    st.sidebar.markdown("Silakan masukkan nilai untuk variabel-variabel di bawah ini.")

    # --- Pilihan Fitur (Sesuai dengan Fitur Model) ---

    # 1. academic_performance (Ordinal: 1-5)
    academic_performance = st.sidebar.select_slider(
        'Kinerja Akademik (academic_performance)',
        options=[1, 2, 3, 4, 5],
        value=3,
        help="Skala 1 (Sangat Buruk) hingga 5 (Sangat Baik)."
    )

    # 2. study_load (Ordinal: 1-5)
    study_load = st.sidebar.select_slider(
        'Beban Belajar (study_load)',
        options=[1, 2, 3, 4, 5],
        value=3,
        help="Skala 1 (Sangat Ringan) hingga 5 (Sangat Berat)."
    )

    # 3. peer_pressure (Ordinal: 1-5)
    peer_pressure = st.sidebar.select_slider(
        'Tekanan Teman Sebaya (peer_pressure)',
        options=[1, 2, 3, 4, 5],
        value=3,
        help="Skala 1 (Sangat Rendah) hingga 5 (Sangat Tinggi)."
    )

    # 4. extracurricular_activities (Ordinal: 0-5)
    extracurricular_activities = st.sidebar.select_slider(
        'Kegiatan Ekstrakurikuler (extracurricular_activities)',
        options=[0, 1, 2, 3, 4, 5],
        value=3,
        help="Skala 0 (Tidak Ada) hingga 5 (Sangat Banyak)."
    )

    # 5. bullying (Ordinal: 1-5)
    bullying = st.sidebar.select_slider(
        'Pengalaman Perundungan (bullying)',
        options=[1, 2, 3, 4, 5],
        value=3,
        help="Skala 1 (Tidak Pernah) hingga 5 (Sangat Sering)."
    )

    # 6. mental_health_history (Biner: 0/1)
    mental_health_history = st.sidebar.radio(
        "Riwayat Kesehatan Mental (mental_health_history)",
        options=[0, 1],
        format_func=lambda x: "Tidak Ada (0)" if x == 0 else "Ada (1)",
        horizontal=True,
        help="Apakah mahasiswa memiliki riwayat masalah kesehatan mental?"
    )

    # --- Persiapan Input Data untuk Model ---

    # Model membutuhkan fitur dalam bentuk DataFrame dengan nama kolom yang spesifik.
    # Catatan: mental_health_history perlu diubah menjadi mental_health_history_1
    input_data = pd.DataFrame({
        'academic_performance': [academic_performance],
        'study_load': [study_load],
        'peer_pressure': [peer_pressure],
        'extracurricular_activities': [extracurricular_activities],
        'bullying': [bullying],
        # Asumsikan 0 menjadi kolom basis, dan 1 menjadi kolom yang ada (seperti OHE)
        'mental_health_history_1': [1 if mental_health_history == 1 else 0]
    })

    # --- Tombol Prediksi ---
    if st.button("Prediksi Tingkat Stres", type="primary"):
        # Lakukan Prediksi
        prediction_score = model.predict(input_data)[0]
        stress_level_result = map_prediction_to_level(prediction_score)

        st.subheader("Hasil Prediksi Tingkat Stres")

        # Visualisasi Hasil
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Skor Prediksi Kontinu",
                value=f"{prediction_score:.4f}",
                help="Skor mentah dari model regresi (nilai antara 0 hingga 2)."
            )

        with col2:
            st.markdown("### Tingkat Stres yang Diprediksi")
            if "Rendah" in stress_level_result:
                st.success(stress_level_result)
            elif "Sedang" in stress_level_result:
                st.warning(stress_level_result)
            else:
                st.error(stress_level_result)
            st.markdown(f"**Interpretasi:** Nilai dibulatkan ke kategori stres terdekat.")

    # --- Bagian Informasi Model (Opsional) ---
    st.markdown("---")
    st.subheader("Informasi Model Regresi Linier")
    st.write(f"Koefisien Model (Bobot Fitur):")

    # Ambil koefisien dan nama fitur dari model yang dimuat
    feature_names = model.feature_names_in_
    coefficients = model.coef_

    # Buat DataFrame untuk menampilkan bobot
    coeff_df = pd.DataFrame({
        'Fitur': feature_names,
        'Koefisien (Bobot)': coefficients
    }).sort_values(by='Koefisien (Bobot)', ascending=False).reset_index(drop=True)

    # Interpretasi: Koefisien Positif berarti fitur tersebut menaikkan skor stres.
    coeff_df['Interpretasi'] = coeff_df['Koefisien (Bobot)'].apply(
        lambda x: "Meningkatkan Stres" if x > 0 else "Menurunkan Stres"
    )

    st.dataframe(coeff_df, use_container_width=True)
    st.write(f"Intercept (Konstanta): **{model.intercept_:.4f}**")
    st.caption("Semakin tinggi nilai koefisien, semakin besar pengaruh positifnya terhadap peningkatan skor stres.")

else:
    st.error("Aplikasi tidak dapat berjalan karena model prediksi gagal dimuat.")

