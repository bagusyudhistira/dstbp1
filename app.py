import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load the trained model ---
try:
    # Memuat model regresi linier yang telah dilatih
    model = joblib.load('linear_regression_model.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'linear_regression_model.pkl' not found. Pastikan file ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Define Expected Model Columns ---
if hasattr(model, 'feature_names_in_'):
    MODEL_EXPECTED_COLUMNS = list(model.feature_names_in_)
else:
    # Kolom fitur yang diharapkan, termasuk hasil one-hot encoding
    MODEL_EXPECTED_COLUMNS = [
        'living_conditions',
        'basic_needs',
        'academic_performance',
        'study_load',
        'social_support',
        'peer_pressure',
        'extracurricular_activities',
        'bullying',
        'mental_health_history_0',        # Untuk 'Tidak Ada'
        'mental_health_history_1'         # Untuk 'Ada'
    ]

# Mapping untuk variabel kategorikal
DUMMY_COLUMN_MAPPING = {
    'Tidak Ada': 'mental_health_history_0',
    'Ada': 'mental_health_history_1'
}

# --- Streamlit App Title ---
st.title('Prediksi Tingkat Stres Mahasiswa')
st.write('Aplikasi untuk memprediksi tingkat stres mahasiswa berdasarkan input yang diberikan.')

# --- Sidebar for User Inputs ---
st.sidebar.header('Input Parameter (Skala 1 - 5)')

def user_input_features():
    # Mengumpulkan input dari user menggunakan slider
    living_conditions = st.sidebar.slider('Kondisi Hidup (1=Buruk, 5=Baik)', 1, 5, 1) # Set default ke 1 (Terburuk)
    basic_needs = st.sidebar.slider('Kebutuhan (1=Sedikit, 5=Banyak)', 1, 5, 1) # Set default ke 1 (Terburuk)
    academic_performance = st.sidebar.slider('Performa Akademik (1=Rendah, 5=Tinggi)', 1, 5, 1) # Set default ke 1 (Terburuk)
    study_load = st.sidebar.slider('Beban Belajar (1=Ringan, 5=Berat)', 1, 5, 5) # Set default ke 5 (Terburuk)
    social_support = st.sidebar.slider('Support Sosial (1=Rendah, 5=Tinggi)', 1, 5, 1) # Set default ke 1 (Terburuk)
    peer_pressure = st.sidebar.slider('Tekanan Teman (1=Rendah, 5=Tinggi)', 1, 5, 5) # Set default ke 5 (Terburuk)
    extracurricular_activities = st.sidebar.slider('Kegiatan Ekstrakurikuler (1=Sedikit, 5=Banyak)', 1, 5, 1) # Set default ke 1 (Terburuk)
    bullying = st.sidebar.slider('Bullying (1=Tidak Ada, 5=Sering)', 1, 5, 5) # Set default ke 5 (Terburuk)
    mental_health_history = st.sidebar.selectbox('Riwayat Kesehatan Mental', ['Tidak Ada', 'Ada'], index=1) # Set default ke 'Ada' (Terburuk)

    data = {
        'living_conditions': living_conditions,
        'basic_needs': basic_needs,
        'academic_performance': academic_performance,
        'study_load': study_load,
        'social_support': social_support,
        'peer_pressure': peer_pressure,
        'extracurricular_activities': extracurricular_activities,
        'bullying': bullying,
        'mental_health_history': mental_health_history
    }
    # Membuat DataFrame dari input pengguna
    return pd.DataFrame(data, index=[0])

# Collect user input
df_input = user_input_features()

# Display the input parameters
st.subheader('Parameter Input Pengguna:')
st.dataframe(df_input, use_container_width=True)

# --- Data Preparation for Prediction ---

# Inisialisasi DataFrame kosong dengan semua kolom yang diharapkan oleh model (diisi nol)
final_input_df = pd.DataFrame(np.zeros((1, len(MODEL_EXPECTED_COLUMNS))), columns=MODEL_EXPECTED_COLUMNS)

# Memasukkan nilai fitur numerik dari input pengguna
for col in ['living_conditions', 'basic_needs','academic_performance', 'study_load', 'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']:
    if col in final_input_df.columns:
        final_input_df[col] = df_input[col][0]

# Menangani variabel kategorikal (mental_health_history) dengan one-hot encoding
mhh_value = df_input['mental_health_history'][0]
dummy_col_name = DUMMY_COLUMN_MAPPING.get(mhh_value)

# Set dummy variable yang relevan menjadi 1
if dummy_col_name and dummy_col_name in final_input_df.columns:
    final_input_df[dummy_col_name] = 1

# --- Prediction Logic ---
if st.sidebar.button('Prediksi Tingkat Stres'):
    try:
        # Lakukan prediksi menggunakan DataFrame yang sudah terstruktur
        prediction = model.predict(final_input_df)
        
        predicted_level_raw = float(prediction[0])

        # **PENTING: CLIPPING**
        # Memastikan hasil prediksi berada dalam rentang skala 1.0 hingga 3.0
        predicted_level_clipped = np.clip(predicted_level_raw, 1.0, 3.0)
        
        # Display predicted stress level
        st.subheader('Hasil Prediksi Tingkat Stres:')
        
        st.markdown(f"**Tingkat Stres diprediksi : Level `{predicted_level_clipped:.2f}`** (Hasil Regresi Murni: `{predicted_level_raw:.2f}`)")

        # Kategori feedback berdasarkan tingkat stres
        if predicted_level_clipped < 1.5:
            st.success("Tingkat Stres Rendah. Pertahankan keseimbangan yang baik!")
        elif predicted_level_clipped < 2.5:
            st.warning("Tingkat Stres Sedang. Perhatikan faktor-faktor pemicu, terutama Beban Belajar atau Tekanan Teman.")
        else: # >= 2.5
            st.error("Tingkat Stres Tinggi. Sangat disarankan untuk segera mencari bantuan profesional (konseling/psikolog).")

    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi. Pastikan semua kolom input sesuai.")
        st.exception(e)

# --- Sidebar Footer ---
st.sidebar.markdown('---')
st.sidebar.markdown('Skala Stres: 1 (Rendah) - 3 (Tinggi)')
st.sidebar.markdown('**Pembaruan:** Hasil prediksi telah dibatasi (clipped) agar tidak melebihi 3.0.')
