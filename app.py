import streamlit as st
import pandas as pd
import numpy as np

st.title("Prediksi Tingkat Stres Mahasiswa - Versi Sederhana dari Nol")
st.write("Aplikasi prediksi tingkat stres mahasiswa dengan model linear buatan sendiri.")

# Fungsi model linear manual dengan koefisien yang sudah saya tetapkan
def simple_stress_model(features):
    """
    features: dict dengan key sesuai input fitur berikut.
    Mengembalikan prediksi skor stres antara 1 sampai 5.
    """
    # Bobot koefisien fitur, beri bobot yang cukup signifikan supaya prediksi menyesuaikan input dan mudah dipahami
    coefs = {
        'living_conditions': 0.5,        # Kondisi hidup buruk naikkan stres
        'basic_needs': 0.6,              # kebutuhan tidak terpenuhi naikkan stres
        'academic_performance': -0.7,    # peforma akademik tinggi turunkan stres
        'study_load': 0.8,               # beban belajar berat naikkan stres
        'social_support': -0.5,          # support sosial tinggi turunkan stres
        'peer_pressure': 0.7,            # tekanan teman naikkan stres
        'extracurricular_activities': 0.3, # ekstrakurikuler banyak naikkan stres sedikit
        'bullying': 1.0,                 # bullying sering naikkan stres banyak
        'mental_health_history': 1.2     # ada riwayat mental naikkan stres cukup banyak
    }
    intercept = 1.0  # nilai minimal stres

    # Hitung prediksi linear manual
    pred = intercept
    for key, coef in coefs.items():
        pred += coef * features[key]

    # Batasi prediksi antara 1 sampai 5
    pred = max(1, min(5, pred))
    return pred

# Input dari user via sidebar
st.sidebar.header("Input Parameter")

living_conditions = st.sidebar.slider("Kondisi Hidup (1=Baik, 5=Buruk)", 1, 5, 3)
basic_needs = st.sidebar.slider("Kebutuhan Tercukupi (1=Baik, 5=Buruk)", 1, 5, 3)
academic_performance = st.sidebar.slider("Performa Akademik (1=Rendah, 5=Tinggi)", 1, 5, 3)
study_load = st.sidebar.slider("Beban Belajar (1=Ringan, 5=Berat)", 1, 5, 3)
social_support = st.sidebar.slider("Support Sosial (1=Rendah, 5=Tinggi)", 1, 5, 3)
peer_pressure = st.sidebar.slider("Tekanan Teman (1=Rendah, 5=Tinggi)", 1, 5, 3)
extracurricular_activities = st.sidebar.slider("Kegiatan Ekstrakurikuler (1=Sedikit, 5=Banyak)", 1, 5, 3)
bullying = st.sidebar.slider("Bullying (1=Tidak Ada, 5=Sering)", 1, 5, 3)
mental_health_history_str = st.sidebar.selectbox("Riwayat Masalah Kesehatan Mental", ['Tidak Ada', 'Ada'])

# Ubah input ke bentuk numerik sesuai kebutuhan model
# Perlu dipastikan skala fitur sesuai
features = {
    'living_conditions': living_conditions,
    'basic_needs': basic_needs,
    'academic_performance': academic_performance,
    'study_load': study_load,
    'social_support': social_support,
    'peer_pressure': peer_pressure,
    'extracurricular_activities': extracurricular_activities,
    'bullying': bullying,
    'mental_health_history': 1 if mental_health_history_str == 'Ada' else 0
}

# Tampilkan input yang diberikan user
st.subheader("Parameter Input Pengguna:")
st.write(pd.DataFrame([features]))

# Jika tombol ditekan, hitung dan tampilkan prediksi
if st.sidebar.button("Prediksi Tingkat Stres"):
    pred = simple_stress_model(features)
    st.subheader("Hasil Prediksi Tingkat Stres:")
    st.markdown(f"**Level Stres diprediksi: {pred:.2f} / 5.00**")

    if pred < 2:
        st.success("Tingkat Stres Rendah")
    elif pred < 3.5:
        st.warning("Tingkat Stres Sedang. Perlu perhatian.")
    else:
        st.error("Tingkat Stres Tinggi. Disarankan mencari bantuan profesional.")

    # Tampilkan bobot koefisien untuk transparansi
    st.subheader("Bobot Fitur Model:")
    for k, v in features.items():
        w = (0.5 if k == 'living_conditions' else
             0.6 if k == 'basic_needs' else
             -0.7 if k == 'academic_performance' else
             0.8 if k == 'study_load' else
             -0.5 if k == 'social_support' else
             0.7 if k == 'peer_pressure' else
             0.3 if k == 'extracurricular_activities' else
             1.0 if k == 'bullying' else
             1.2)
        st.write(f"{k}: bobot {w}, nilai input {v}, kontribusi: {w*v:.2f}")

    st.write(f"Intercept model: 1.0")

