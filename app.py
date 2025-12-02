import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys

# --- Pemetaan Skala Ordinal (1-5) ---
# Ini digunakan untuk menampilkan label yang mudah dipahami kepada pengguna
RATING_MAP = {
    1: "1 - Sangat Rendah/Buruk",
    2: "2 - Rendah",
    3: "3 - Sedang/Normal",
    4: "4 - Tinggi",
    5: "5 - Sangat Tinggi/Baik"
}

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Tingkat Stres Mahasiswa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Pemetaan Kategori Hasil ---
def map_prediction_to_level(score):
    """Memetakan skor prediksi kontinyu ke kategori tingkat stres (0, 1, 2)."""
    # Batas 0.5 dan 1.5 digunakan untuk memetakan skor kontinu ke kategori diskrit 0, 1, atau 2.
    if score < 0.5:
        return "Tingkat Stres Rendah (0)"
    elif score < 1.5:
        return "Tingkat Stres Sedang (1)"
    else:
        return "Tingkat Stres Tinggi (2)"

# --- Memuat Model ---
@st.cache_resource
def load_model(file_path):
    """Memuat model dari file .pkl dengan penanganan error yang lebih baik."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            
        # Pengecekan versi scikit-learn
        try:
            from sklearn import __version__ as sk_version
            st.sidebar.caption(f"Versi scikit-learn yang digunakan: {sk_version}")
        except ImportError:
            st.sidebar.caption("Tidak dapat memeriksa versi scikit-learn.")
            
        return model
    except Exception as e:
        # Jika ada error saat memuat, tampilkan pesan yang spesifik
        st.error(f"⚠️ Gagal memuat model '{file_path}'. Ini sering terjadi karena ketidakcocokan versi pustaka (terutama scikit-learn).")
        st.error(f"Detail Error: {e}")
        return None

MODEL_PATH = "linear_regression_model (3).pkl"
model = load_model(MODEL_PATH)

# --- Judul dan Deskripsi Aplikasi ---
st.title("🧠 Prediktor Tingkat Stres Mahasiswa")
st.markdown("""
Aplikasi ini memprediksi tingkat stres (Rendah, Sedang, Tinggi) berdasarkan beberapa faktor utama
menggunakan Model Regresi Linier yang telah dilatih dari data survei.
""")

if model is not None:
    
    # Dapatkan nama fitur yang diharapkan model
    try:
        FEATURE_NAMES = model.feature_names_in_.tolist()
    except AttributeError:
        # Fallback jika model tidak menyimpan feature_names_in_ (untuk versi scikit-learn lama)
        st.warning("Nama fitur tidak tersedia dalam metadata model. Menggunakan urutan default.")
        FEATURE_NAMES = ['academic_performance', 'study_load', 'peer_pressure', 
                         'extracurricular_activities', 'bullying', 'mental_health_history_1']
        
    st.sidebar.header("Input Fitur Mahasiswa")
    st.sidebar.markdown("Silakan masukkan nilai untuk variabel-variabel di bawah ini.")

    # --- Kolom Input Variabel Ordinal/Skala ---
    
    # Fungsi untuk membuat slider input dengan label teks
    def create_labeled_slider(label, key, help_text, options, default_value, rating_map):
        return st.sidebar.select_slider(
            label,
            options=options,
            value=default_value,
            key=key,
            help=help_text,
            # Format_func akan menampilkan teks, tetapi mengembalikan nilai integer
            format_func=lambda x: rating_map.get(x, str(x))
        )

    # Catatan: Kita menggunakan daftar opsi [1, 2, 3, 4, 5] untuk mendapatkan nilai integer.
    
    academic_performance = create_labeled_slider(
        'Kinerja Akademik', 'ap', "Skala 1 (Sangat Buruk) hingga 5 (Sangat Baik).",
        options=[1, 2, 3, 4, 5], default_value=3, rating_map=RATING_MAP
    )
    study_load = create_labeled_slider(
        'Beban Belajar', 'sl', "Skala 1 (Sangat Ringan) hingga 5 (Sangat Berat).",
        options=[1, 2, 3, 4, 5], default_value=3, rating_map=RATING_MAP
    )
    peer_pressure = create_labeled_slider(
        'Tekanan Teman Sebaya', 'pp', "Skala 1 (Sangat Rendah) hingga 5 (Sangat Tinggi).",
        options=[1, 2, 3, 4, 5], default_value=3, rating_map=RATING_MAP
    )
    
    # Perlakuan khusus untuk extracurricular_activities yang dimulai dari 0
    extracurricular_options = {
        0: "0 - Tidak Ada", 1: "1 - Sangat Rendah", 2: "2 - Rendah", 
        3: "3 - Sedang", 4: "4 - Tinggi", 5: "5 - Sangat Banyak"
    }
    extracurricular_activities = st.sidebar.select_slider(
        'Kegiatan Ekstrakurikuler',
        options=[0, 1, 2, 3, 4, 5],
        value=3,
        key='ea',
        help="Skala 0 (Tidak Ada) hingga 5 (Sangat Banyak).",
        format_func=lambda x: extracurricular_options.get(x, str(x))
    )
    
    bullying = create_labeled_slider(
        'Pengalaman Perundungan', 'b', "Skala 1 (Tidak Pernah) hingga 5 (Sangat Sering).",
        options=[1, 2, 3, 4, 5], default_value=3, rating_map=RATING_MAP
    )

    # --- Input Variabel Biner ---
    mental_health_history = st.sidebar.radio(
        "Riwayat Kesehatan Mental",
        options=[0, 1],
        format_func=lambda x: "Tidak Ada (0)" if x == 0 else "Ada (1)",
        horizontal=True,
        key='mhh',
        help="Apakah mahasiswa memiliki riwayat masalah kesehatan mental (1=Ya, 0=Tidak)?"
    )

    # --- Persiapan Input Data untuk Model ---
    
    input_values = {
        'academic_performance': academic_performance,
        'study_load': study_load,
        'peer_pressure': peer_pressure,
        'extracurricular_activities': extracurricular_activities,
        'bullying': bullying,
        # OHE: 1 jika ada riwayat, 0 jika tidak
        'mental_health_history_1': 1 if mental_health_history == 1 else 0
    }
    
    # Pastikan data diurutkan sesuai dengan kebutuhan model (FEATURE_NAMES)
    ordered_input = {name: [input_values[name]] for name in FEATURE_NAMES}
    input_df = pd.DataFrame(ordered_input)
    
    # --- Tombol Prediksi ---
    if st.button("Prediksi Tingkat Stres", type="primary"):
        # Lakukan Prediksi
        try:
            prediction_score = model.predict(input_df)[0]
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
                # Gunakan CSS kustom untuk memberikan highlight yang lebih baik
                if "Rendah" in stress_level_result:
                    st.success(stress_level_result)
                elif "Sedang" in stress_level_result:
                    st.warning(stress_level_result)
                else:
                    st.error(stress_level_result)
                st.markdown(f"**Interpretasi:** Nilai dibulatkan ke kategori stres terdekat (0=Rendah, 2=Tinggi).")

        except Exception as e:
            st.error("Terjadi error saat menjalankan prediksi. Pastikan semua input sudah benar.")
            st.error(f"Error: {e}")

    # --- Bagian Informasi Model ---
    st.markdown("---")
    st.subheader("Informasi Bobot Fitur (Koefisien Regresi)")
    
    # Ambil koefisien dan nama fitur dari model yang dimuat
    coefficients = model.coef_
    
    coeff_df = pd.DataFrame({
        'Fitur': FEATURE_NAMES,
        'Koefisien (Bobot)': coefficients
    }).sort_values(by='Koefisien (Bobot)', ascending=False).reset_index(drop=True)
    
    # Interpretasi: Koefisien Positif berarti fitur tersebut menaikkan skor stres.
    coeff_df['Interpretasi'] = coeff_df['Koefisien (Bobot)'].apply(
        lambda x: "Meningkatkan Stres" if x > 0 else "Menurunkan Stres"
    )
    
    st.dataframe(coeff_df, use_container_width=True)
    st.write(f"Intercept (Konstanta Model): **{model.intercept_:.4f}**")
    st.caption("Semakin besar nilai koefisien positif, semakin besar pengaruh fitur tersebut dalam memprediksi tingkat stres yang lebih tinggi.")

else:
    # Tampilkan pesan ini jika load_model gagal
    st.error("Aplikasi tidak dapat menampilkan input atau prediksi karena model gagal dimuat.")
