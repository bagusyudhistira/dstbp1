import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load the trained model ---
try:
    model = joblib.load('linear_regression_model.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'linear_regression_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Define Expected Model Columns ---
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
        'mental_health_history_0',       # For 'Tidak Ada'
        'mental_health_history_1'        # For 'Ada'
    ]

# Mapping for categorical variables to expected dummy columns
DUMMY_COLUMN_MAPPING = {
    'Tidak Ada': 'mental_health_history_0',
    'Ada': 'mental_health_history_1'
}

# --- Streamlit App Title ---
st.title('Prediksi Tingkat Stres Mahasiswa')
st.write('Aplikasi untuk memprediksi tingkat stres mahasiswa berdasarkan input yang diberikan.')

# --- Sidebar for User Inputs ---
st.sidebar.header('Input Parameter')

def user_input_features():
    living_conditions = st.sidebar.slider('Kondisi Hidup (1=Buruk, 5=Baik)', 1, 5, 3)
    basic_needs = st.sidebar.slider('Kebutuhan (1=Sedikit, 5=Banyak)', 1, 5, 3)
    academic_performance = st.sidebar.slider('Performa Akademik (1=Rendah, 5=Tinggi)', 1, 5, 3)
    study_load = st.sidebar.slider('Beban Belajar (1=Ringan, 5=Berat)', 1, 5, 3)
    social_support = st.sidebar.slider('Support Sosial (1=Rendah, 5=Tinggi)', 1, 5, 3)
    peer_pressure = st.sidebar.slider('Tekanan Teman (1=Rendah, 5=Tinggi)', 1, 5, 3)
    extracurricular_activities = st.sidebar.slider('Kegiatan Ekstrakurikuler (1=Sedikit, 5=Banyak)', 1, 5, 3)
    bullying = st.sidebar.slider('Bullying (1=Tidak Ada, 5=Sering)', 1, 5, 3)
    mental_health_history = st.sidebar.selectbox('Riwayat Kesehatan Mental', ['Tidak Ada', 'Ada'])

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

# Collect user input
df_input = user_input_features()

# Display the input parameters
st.subheader('Parameter Input Pengguna:')
st.dataframe(df_input, use_container_width=True)

# --- Data Preparation for Prediction ---

# Create an empty DataFrame initialized with zeros, with the exact columns expected by the model
final_input_df = pd.DataFrame(np.zeros((1, len(MODEL_EXPECTED_COLUMNS))), columns=MODEL_EXPECTED_COLUMNS)

# Populate numerical features using the values from the user input DataFrame (df_input)
for col in ['living_conditions', 'basic_needs','academic_performance', 'study_load', 'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']:
    if col in final_input_df.columns:
        final_input_df[col] = df_input[col][0]

# Handle the categorical variable (mental_health_history) and convert to one-hot encoding
mhh_value = df_input['mental_health_history'][0]
dummy_col_name = DUMMY_COLUMN_MAPPING.get(mhh_value)

# Set the relevant dummy variable to 1
if dummy_col_name and dummy_col_name in final_input_df.columns:
    final_input_df[dummy_col_name] = 1

# --- Prediction Logic ---
if st.sidebar.button('Prediksi Tingkat Stres'):
    try:
        # Perform prediction using the structured final_input_df
        prediction = model.predict(final_input_df)
        
        predicted_level = float(prediction[0])  # Ensure the prediction is a float

        # Display predicted stress level
        st.subheader('Hasil Prediksi Tingkat Stres:')
        st.markdown(f"**Tingkat Stres diprediksi : Level `{predicted_level:.2f}`**")

        # Provide feedback based on the predicted level
        if predicted_level < 1:
            st.success("Tingkat Stres Rendah.")
        elif predicted_level < 2:
            st.warning("Tingkat Stres Sedang. Perlu perhatian.")
        else:
            st.error("Tingkat Stres Tinggi. Sangat disarankan untuk mencari bantuan.")

    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi. Pastikan semua kolom input sesuai.")
        st.exception(e)

# --- Sidebar Footer ---
st.sidebar.markdown('---')
st.sidebar.markdown('Skala Stres: 1 (Rendah) - 3 (Tinggi)')
