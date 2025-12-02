import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
# IMPORTANT: Pastikan 'linear_regression_model.pkl' ditempatkan dengan benar dan disimpan dengan joblib.
try:
    model = joblib.load('linear_regression_model.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'linear_regression_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Mengambil kolom yang diharapkan dari model secara definitif ---
# Prioritas: 1. Kolom dari model. 2. Kolom fallback (menggunakan _0 dan _1).
if hasattr(model, 'feature_names_in_'):
    # Mengambil daftar kolom yang digunakan saat training dari atribut model (cara paling aman)
    MODEL_EXPECTED_COLUMNS = list(model.feature_names_in_)
else:
    # Fallback ke nama kolom yang ditunjukkan oleh error traceback (_0 dan _1)
    MODEL_EXPECTED_COLUMNS = [
        'academic_performance',
        'study_load',
        'peer_pressure',
        'extracurricular_activities',
        'bullying',
        'mental_health_history_0',       # Diperkirakan untuk 'Tidak Ada'
        'mental_health_history_1'       # Diperkirakan untuk 'Ada'
    ]
    # NOTE: Jika model Anda menggunakan nama kolom 'mental_health_history_Ada' dan 'mental_health_history_Tidak Ada',
    # Anda harus mengganti baris di atas secara manual agar sesuai.

# Mapping untuk konversi pilihan pengguna ke kolom dummy yang diharapkan model
# Ini harus sinkron dengan MODEL_EXPECTED_COLUMNS
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
else:
    st.error(f"Peringatan: Tidak dapat mencocokkan kolom dummy untuk Riwayat Mental '{mhh_value}'. Cek kembali model Anda. Kolom yang diharapkan: 'mental_health_history_0' dan 'mental_health_history_1'.")


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
        st.error("Terjadi kesalahan saat melakukan prediksi. Ada masalah dengan struktur data yang diharapkan model.")
        st.exception(e)

st.sidebar.markdown('---')
st.sidebar.markdown('Skala Stres: 1 (Sangat Rendah) - 5 (Sangat Tinggi)')
