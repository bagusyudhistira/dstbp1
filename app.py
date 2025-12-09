import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('linear_regression_model.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'linear_regression_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Try to load scaler if used during training
try:
    scaler = joblib.load('scaler.pkl')
    scaler_available = True
except FileNotFoundError:
    scaler_available = False
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    scaler_available = False

# Determine expected columns from model or fallback columns
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
        'mental_health_history_0',  # Tidak Ada
        'mental_health_history_1'   # Ada
    ]

# Mapping for dummy variables of mental health history
DUMMY_COLUMN_MAPPING = {
    'Tidak Ada': 'mental_health_history_0',
    'Ada': 'mental_health_history_1'
}

st.title('Prediksi Tingkat Stres Mahasiswa')
st.write('Aplikasi untuk memprediksi tingkat stres mahasiswa.')

st.sidebar.header('Input Parameter')

def user_input_features():
    living_conditions = st.sidebar.slider('Kondisi Hidup (1=Rendah, 5=Tinggi)', 1, 5, 3)
    basic_needs = st.sidebar.slider('Kebutuhan (1=Rendah, 5=Tinggi)', 1, 5, 3)
    academic_performance_input = st.sidebar.slider('Peforma Akademik (1=Rendah, 5=Tinggi)', 1, 5, 3)
    # Membalik skala academic_performance supaya sesuai dengan model (koef negatif)
    academic_performance = 6 - academic_performance_input
    study_load = st.sidebar.slider('Beban Belajar (1=Ringan, 5=Berat)', 1, 5, 3)
    social_support = st.sidebar.slider('Support Sosial (1=Ringan, 5=Berat)', 1, 3, 2)
    peer_pressure = st.sidebar.slider('Tekanan Teman (1=Rendah, 5=Tinggi)', 1, 5, 3)
    extracurricular_activities = st.sidebar.slider('Kegiatan Ekstrakurikuler (1=Sedikit, 5=Banyak)', 1, 5, 3)
    bullying = st.sidebar.slider('Bullying (1=Tidak Ada, 5=Sering)', 1, 5, 3)
    mental_health_history = st.sidebar.selectbox('Riwayat Mental', ['Tidak Ada', 'Ada'])

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

st.subheader('Parameter Input Pengguna:')
st.dataframe(df_input, use_container_width=True)

# Prepare data frame with expected columns, filled by zeros
final_input_df = pd.DataFrame(np.zeros((1, len(MODEL_EXPECTED_COLUMNS))), columns=MODEL_EXPECTED_COLUMNS)

# Set numerical inputs
for col in ['living_conditions', 'basic_needs', 'academic_performance', 'study_load', 'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']:
    if col in final_input_df.columns:
        final_input_df[col] = df_input[col][0]

# Handle dummy variable mental_health_history
mhh_value = df_input['mental_health_history'][0]
dummy_col_name = DUMMY_COLUMN_MAPPING.get(mhh_value)
if dummy_col_name and dummy_col_name in final_input_df.columns:
    final_input_df[dummy_col_name] = 1
else:
    st.error(f"Peringatan: Tidak dapat mencocokkan kolom dummy untuk Riwayat Mental '{mhh_value}'.")

# Scale input if scaler available
if scaler_available:
    scaled_input = scaler.transform(final_input_df)
else:
    scaled_input = final_input_df.values

# Fungsi untuk melakukan scaling ulang prediksi agar output di range 1-5
def scale_prediction_to_1_5(pred_raw, pred_min=1.5, pred_max=3.0):
    # Linear scaling prediksi pred_raw ke rentang 1-5
    scaled = 1 + (pred_raw - pred_min) * 4 / (pred_max - pred_min)
    scaled = max(1, min(5, scaled))
    return scaled

if st.sidebar.button('Prediksi Tingkat Stres'):
    try:
        prediction_raw = model.predict(scaled_input)[0]
        prediction_scaled = scale_prediction_to_1_5(prediction_raw)

        st.subheader('Hasil Prediksi Tingkat Stres:')
        st.markdown(f"**Tingkat Stres diprediksi : Level `{prediction_scaled:.2f}`**")

        if prediction_scaled < 2:
            st.success("Tingkat Stres Rendah.")
        elif prediction_scaled < 3.5:
            st.warning("Tingkat Stres Sedang. Perlu perhatian.")
        else:
            st.error("Tingkat Stres Tinggi. Sangat disarankan untuk mencari bantuan.")

        st.subheader("Koefisien Model (Feature Coefficients):")
        for feature, coef in zip(MODEL_EXPECTED_COLUMNS, model.coef_):
            st.write(f"{feature}: {coef:.3f}")
        st.write(f"Intercept: {model.intercept_:.3f}")

        manual_pred_raw = np.dot(final_input_df.iloc[0], model.coef_) + model.intercept_
        st.write(f"Perhitungan manual prediksi (tanpa scaling): {manual_pred_raw:.3f}")

    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi. Pastikan semua kolom input sesuai.")
        st.exception(e)

st.sidebar.markdown('---')
st.sidebar.markdown('Skala Stres: 1 (Sangat Rendah) - 5 (Sangat Tinggi)')
