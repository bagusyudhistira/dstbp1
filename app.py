import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dari file .pkl (pastikan file berada di folder yang sama dengan app)
try:
    model = joblib.load('linear_regression_model.pkl')
except FileNotFoundError:
    st.error("File model 'linear_regression_model.pkl' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Error saat load model: {e}")
    st.stop()

# Mapping dummy untuk fitur kategorikal mental_health_history sesuai model
DUMMY_COLUMN_MAPPING = {
    'Tidak Ada': 'mental_health_history_0',
    'Ada': 'mental_health_history_1'
}

if hasattr(model, 'feature_names_in_'):
    EXPECTED_COLUMNS = list(model.feature_names_in_)
else:
    EXPECTED_COLUMNS = [
        'living_conditions',
        'basic_needs',
        'academic_performance',
        'study_load',
        'social_support',
        'peer_pressure',
        'extracurricular_activities',
        'bullying',
        'mental_health_history_0',
        'mental_health_history_1'
    ]

st.title("Prediksi Tingkat Stres Mahasiswa")

# Sidebar untuk input pengguna
st.sidebar.header("Masukkan Parameter Input")

def user_input():
    data = {
        'living_conditions': st.sidebar.slider('Kebutuhan Tercukupi (1=Baik, 5=Buruk)', 1, 5, 3),
        'basic_needs': st.sidebar.slider('Peforma Akademik (1=Rendah, 5=Tinggi)', 1, 5, 3),
        'academic_performance': st.sidebar.slider('Performa Akademik (1=Rendah, 5=Tinggi)', 1, 5, 3),
        'study_load': st.sidebar.slider('Beban Belajar (1=Ringan, 5=Berat)', 1, 5, 3),
        'social_support': st.sidebar.slider('Support Sosial (1=Rendah, 5=Tinggi)', 1, 5, 3),
        'peer_pressure': st.sidebar.slider('Tekanan Teman (1=Rendah, 5=Tinggi)', 1, 5, 3),
        'extracurricular_activities': st.sidebar.slider('Kegiatan Ekstrakurikuler (1=Sedikit, 5=Banyak)', 1, 5, 3),
        'bullying': st.sidebar.slider('Bullying (1=Tidak Ada, 5=Sering)', 1, 5, 3),
        'mental_health_history': st.sidebar.selectbox('Riwayat Masalah Kesehatan Mental', ['Tidak Ada', 'Ada'])
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input()

st.subheader("Parameter Input Pengguna:")
st.dataframe(df_input)

# Persiapkan dataframe input untuk model sesuai kolom yang diharapkan, inisialisasi 0
final_input = pd.DataFrame(np.zeros((1, len(EXPECTED_COLUMNS))), columns=EXPECTED_COLUMNS)

# Isi fitur numerik
numeric_cols = ['living_conditions', 'basic_needs', 'academic_performance', 'study_load',
                'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']

for col in numeric_cols:
    if col in final_input.columns:
        final_input[col] = df_input[col].values[0]

# Isi fitur dummy mental_health_history
dummy_col = DUMMY_COLUMN_MAPPING.get(df_input['mental_health_history'].values[0])
if dummy_col in final_input.columns:
    final_input[dummy_col] = 1
else:
    st.error(f"Kolom dummy '{dummy_col}' tidak ditemukan di model.")

# Tombol prediksi
if st.sidebar.button("Prediksi Tingkat Stres"):
    try:
        pred = model.predict(final_input)[0]

        # Tentukan range prediksi linear untuk scaling manual
        pred_min = 1.0   # estimasi minimal prediksi dari data training (sesuaikan jika perlu)
        pred_max = 9.2   # estimasi maksimal prediksi dari kombinasi input terbesar (ubah sesuai model Anda)

        # Fungsi scaling prediksi linear ke range 1-5 agar proporsional
        def scale_prediction(pred_raw, min_pred, max_pred):
            scaled = 1 + ((pred_raw - min_pred) * 4) / (max_pred - min_pred)
            return max(1, min(5, scaled))

        pred_scaled = scale_prediction(pred, pred_min, pred_max)

        st.subheader("Hasil Prediksi Tingkat Stres:")
        st.markdown(f"**Level Stres diprediksi (skala 1-5): {pred_scaled:.2f}**")

        if pred_scaled < 2:
            st.success("Tingkat Stres Rendah")
        elif pred_scaled < 3.5:
            st.warning("Tingkat Stres Sedang. Perlu Perhatian.")
        else:
            st.error("Tingkat Stres Tinggi. Disarankan mencari bantuan profesional.")

        # Tampilkan bobot model dan kontribusi tiap fitur jika tersedia
        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            st.subheader("Bobot Fitur Model dan Kontribusi:")
            for feat, coef in zip(EXPECTED_COLUMNS, model.coef_):
                val = final_input[feat].values[0]
                contrib = coef * val
                st.write(f"{feat}: bobot {coef:.3f}, nilai {val}, kontribusi {contrib:.3f}")
            st.write(f"Intercept model: {model.intercept_:.3f}")

    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi.")
        st.exception(e)
