import streamlit as st
import pandas as pd
import pickle
import joblib

# Load the trained model
model = joblib.load('linear_regression_model.pkl') # Changed model filename and loading method

# Streamlit app title
st.title('Prediksi Tingkat Stres Mahasiswa')
st.write('Aplikasi untuk memprediksi tingkat stres mahasiswa.')

# Sidebar for user inputs
st.sidebar.header('Input Parameter')

def user_input_features():
    academic_performance = st.sidebar.slider('Peforma Akademik', 1, 2, 3, 4, 5)
    study_load = st.sidebar.slider('Beban Belajar', 1, 2, 3, 4, 5)
    peer_pressure = st.sidebar.slider('Tekanan Teman', 1, 2, 3, 4, 5)
    extracurricular_activities = st.sidebar.slider('Kegiatan Ekstrakurikuler', 1, 2, 3, 4, 5)
    bullying = st.sidebar.slider('Peforma Akademik', 1, 2, 3, 4, 5)

    mental_health_history = st.sidebar.selectbox('Riwayat Mental', ['Ada', 'Tidak Ada'])

    data = {
        'academic_performance': academic_performance,
        'study_load': study_load,
        'peer_pressure': peer_pressure,
        'extracurricular_activities': extracurricular_activities,
        'bullying': bullying,
        'mental_health_history':mental_health_history
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('Parameter Input Pengguna:')
st.write(df_input)

# Define the exact columns and their dtypes expected by the model during training
# This list ensures correct order and includes all dummy variables used during training
training_columns_and_dtypes = {
    'academic_performance': 'int64',
    'study_load': 'int64',
    'peer_pressure': 'int64',
    'extracurricular_activities': 'int64',
    'bullying': 'int64',
    'mental_health_history': 'bool'
}

# Create an empty DataFrame with the correct columns and dtypes
final_input_df = pd.DataFrame(columns=training_columns_and_dtypes.keys())
for col, dtype in training_columns_and_dtypes.items():
    final_input_df[col] = final_input_df[col].astype(dtype)

# Add a single row of data, initially all zeros/False
final_input_df.loc[0] = 0
for col, dtype in training_columns_and_dtypes.items():
    if dtype == 'bool':
        final_input_df.loc[0, col] = False

# Populate numerical features
final_input_df.loc[0, 'academic_performance'] = df_input['academic_performance'][0]
final_input_df.loc[0, 'study_load'] = df_input['study_load'][0]
final_input_df.loc[0, 'peer_pressure'] = df_input['peer_pressure'][0]
final_input_df.loc[0, 'extracurricular_activities'] = df_input['extracurricular_activities'][0]
final_input_df.loc[0, 'bullying'] = df_input['bullying'][0]

# Populate one-hot encoded categorical features
selected_mhh_col = f"mental_health_history_{df_input['mental_health_history'][0]}"
if selected_mhh_col in final_input_df.columns:
    final_input_df.loc[0, selected_mhh_col] = True

# Make prediction
if st.sidebar.button('Prediksi Tagihan'):
    try:
        prediction = model.predict(final_input_df) # Use the new DataFrame name
        st.subheader('Hasil Tingkat Stres:')
        st.write(f"Tingkat Stress diprediksi : Level {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.exception(e) # Show full traceback
