import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- KONFIGURASI NAMA FILE MODEL ---
# GANTI NAMA FILE DI SINI JIKA ANDA MENGGUNAKAN MODEL BARU (Contoh: 'random_forest_model.pkl')
MODEL_FILENAME = 'rfr_model.pkl'

# Load the trained model
try:
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_FILENAME}' not found. Pastikan file tersebut ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- BLOK PENYESUAIAN NAMA KOLOM SECARA DEFINITIF ---
# Berdasarkan error traceback yang berulang, model HANYA menerima format _0 dan _1.
# Kami menerapkan solusi yang diminta model.

# Kolom yang DILIHAT oleh model saat training (DIPAKSA menggunakan _0 dan _1)
MODEL_EXPECTED_COLUMNS = [
    'academic_performance', 
    'study_load', 
    'peer_pressure', 
    'extracurricular_activities', 
    'bullying',
    'mental_health_history_0',       # Kolom untuk 'Tidak Ada'
    'mental_health_history_1'        # Kolom untuk 'Ada'
]

# Mapping yang Sesuai dengan Kolom Model
DUMMY_COLUMN_MAPPING = {
    'Tidak Ada': 'mental_health_history_0',
    'Ada': 'mental_health_history_1'
}


# Streamlit app title
st.title('Prediksi Tingkat Stres Mahasiswa')
st.write('Aplikasi untuk memprediksi tingkat stres mahasiswa.')

# Sidebar for user inputs
st.sidebar.header('Input Parameter')

def user_input_features():
    # Sliders and selectbox for collecting user input
    academic_performance = st.sidebar.slider('Peforma Akademik (1=Rendah, 5=Tinggi)', 1, 5, 3)
    study_load = st.sidebar.slider('Beban Belajar (1=Ringan, 5=Berat)', 1, 5, 3)
    peer_pressure = st.sidebar.slider('Tekanan Teman (1=Rendah, 5=Tinggi)', 1, 5, 3) 
    extracurricular_activities = st.sidebar.slider('Kegiatan Ekstrakurikuler (1=Sedikit, 5=Banyak)', 1, 5, 3)
    bullying = st.sidebar.slider('Bullying (1=Tidak Ada, 5=Sering)', 1, 5, 3)
    mental_health_history = st.sidebar.selectbox('Riwayat Mental', ['Tidak Ada', 'Ada'])

    data = {
        'academic_performance': academic_performance,
        'study_load': study_load,
        'peer_pressure': peer_pressure,
        'extracurricular_activities': extracurricular_activities,
        'bullying': bullying,
        'mental_health_history': mental_health_history
    }
    # Return the data as a simple DataFrame for display purposes
    return pd.DataFrame(data, index=[0])

# Collect user input
df_input = user_input_features()

st.subheader('Parameter Input Pengguna:')
st.dataframe(df_input, use_container_width=True)


# --- Data Preparation for Prediction ---

# Membuat DataFrame dengan semua kolom model yang diharapkan, diisi nol
final_input_df = pd.DataFrame(np.zeros((1, len(MODEL_EXPECTED_COLUMNS))), columns=MODEL_EXPECTED_COLUMNS)

# Populate numerical features
for col in ['academic_performance', 'study_load', 'peer_pressure', 'extracurricular_activities', 'bullying']:
    if col in final_input_df.columns:
        final_input_df[col] = df_input[col][0]

# Populate the one-hot encoded categorical feature
mhh_value = df_input['mental_health_history'][0]
dummy_col_name = DUMMY_COLUMN_MAPPING.get(mhh_value)

# Set the relevant dummy variable to 1
if dummy_col_name and dummy_col_name in final_input_df.columns:
    final_input_df[dummy_col_name] = 1
# else block yang menampilkan error telah dihapus untuk menghindari false positive

# --- DEBUGGING: Tampilkan kolom yang akan diprediksi ---
st.markdown("---")
st.caption(f"Kolom yang dikirim ke model untuk prediksi: {final_input_df.columns.tolist()}")
st.markdown("---")

# Make prediction
if st.sidebar.button('Prediksi Tingkat Stres'):
    try:
        # PENTING: final_input_df sudah memiliki urutan kolom yang sama dengan MODEL_EXPECTED_COLUMNS
        prediction = model.predict(final_input_df)
        
        # Ensure prediction is a float and format the result
        predicted_level = float(prediction[0])
        
        st.subheader('Hasil Prediksi Tingkat Stres:')
        st.markdown(f"**Tingkat Stres diprediksi : Level `{predicted_level:.2f}`**")

        if predicted_level < 2:
            st.success("Tingkat Stres Rendah.")
        elif predicted_level < 3.5:
            st.warning("Tingkat Stres Sedang. Perlu perhatian.")
        else:
            st.error("Tingkat Stres Tinggi. Sangat disarankan untuk mencari bantuan.")

    except Exception as e:
        # Jika error terjadi di sini, itu karena masalah pada model, bukan lagi pada dataframe input.
        st.error("Terjadi kesalahan saat melakukan prediksi. Pastikan semua kolom input (terutama nama kolom) sudah sesuai dengan model yang disimpan.")
        st.exception(e)

st.sidebar.markdown('---')
st.sidebar.markdown('Skala Stres: 1 (Sangat Rendah) - 5 (Sangat Tinggi)')
