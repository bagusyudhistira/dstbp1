import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model from .pkl file (pastikan file 'linear_regression_model.pkl' sudah ada di folder yang sama)
try:
    model = joblib.load('linear_regression_model.pkl')
except FileNotFoundError:
    st.error("Model 'linear_regression_model.pkl' tidak ditemukan. Pastikan file model ada di folder yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

# Mapping kolom dummy untuk fitur mental_health_history sesuai model
DUMMY_COLUMN_MAPPING = {
    'Tidak Ada': 'mental_health_history_0',
    'Ada': 'mental_health_history_1'
}

if hasattr(model, 'feature_names_in_'):
    MODEL_EXPECTED_COLUMNS = list(model.feature_names_in_)
else:
    MODEL_EXPECTED_COLUMNS = [
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
st.write("Masukkan parameter berikut untuk memprediksi tingkat stres.")

# Sidebar input parameter
st.sidebar.header("Input Parameter")

def user_input_features():
    living_conditions = st.sidebar.slider("Kebutuhan Tercukupi (1=Baik, 5=Buruk)", 1, 5, 3)
    basic_needs = st.sidebar.slider("Performa Akademik (1=Rendah, 5=Tinggi)", 1, 5, 3)
    study_load = st.sidebar.slider("Beban Belajar (1=Ringan, 5=Berat)", 1, 5, 3)
    social_support = st.sidebar.slider("Support Sosial (1=Rendah, 5=Tinggi)", 1, 5, 3)
    peer_pressure = st.sidebar.slider("Tekanan Teman (1=Rendah, 5=Tinggi)", 1, 5, 3)
    extracurricular_activities = st.sidebar.slider("Kegiatan Ekstrakurikuler (1=Sedikit, 5=Banyak)", 1, 5, 3)
    bullying = st.sidebar.slider("Bullying (1=Tidak Ada, 5=Sering)", 1, 5, 3)
    mental_health_history = st.sidebar.selectbox("Riwayat Masalah Kesehatan Mental", ['Tidak Ada', 'Ada'])

    academic_performance = st.sidebar.slider("Performa Akademik (1=Rendah, 5=Tinggi)", 1, 5, 3)
    
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
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()
st.subheader("Parameter Input Pengguna:")
st.dataframe(df_input, use_container_width=True)

# Menyiapkan input ke model sesuai kolom yang diharapkan
final_input_df = pd.DataFrame(np.zeros((1, len(MODEL_EXPECTED_COLUMNS))), columns=MODEL_EXPECTED_COLUMNS)

# Mengisi fitur numerik
numerical_features = [
    'living_conditions', 'basic_needs', 'academic_performance', 'study_load',
    'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying'
]
for col in numerical_features:
    if col in final_input_df.columns:
        final_input_df[col] = df_input[col].values[0]

# Mengisi dummy fitur mental_health_history
dummy_col_name = DUMMY_COLUMN_MAPPING.get(df_input['mental_health_history'].values[0])
if dummy_col_name in final_input_df.columns:
    final_input_df[dummy_col_name] = 1
else:
    st.error(f"Kolom dummy '{dummy_col_name}' tidak ditemukan di model.")

# Tombol prediksi
if st.sidebar.button("Prediksi Tingkat Stres"):
    try:
        prediction = model.predict(final_input_df)[0]
        prediction = max(1, min(5, prediction))  # Batasi prediksi di antara 1-5

        st.subheader("Hasil Prediksi Tingkat Stres:")
        st.markdown(f"**Level Stres diprediksi: {prediction:.2f} / 5.00**")
        if prediction < 2:
            st.success("Tingkat Stres Rendah")
        elif prediction < 3.5:
            st.warning("Tingkat Stres Sedang. Perlu perhatian.")
        else:
            st.error("Tingkat Stres Tinggi. Disarankan mencari bantuan profesional.")

        # Tampilkan koefisien model jika ada
        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            st.subheader("Bobot Fitur Model:")
            for f, c in zip(MODEL_EXPECTED_COLUMNS, model.coef_):
                val = final_input_df[f].values[0]
                st.write(f"{f}: bobot {c:.3f}, nilai input {val}, kontribusi {c*val:.3f}")
            st.write(f"Intercept model: {model.intercept_:.3f}")

    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi.")
        st.exception(e)
