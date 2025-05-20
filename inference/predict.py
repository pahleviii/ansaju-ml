import os
import joblib
import numpy as np

# Path absolut dinamis berdasarkan posisi file predict.py
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MODEL_PATH = os.path.join(BASE_DIR, 'model_ansaju.pkl')
FEATURE_PATH = os.path.join(BASE_DIR, 'fitur_model.pkl')

# Load model
try:
    model = joblib.load(MODEL_PATH)
    fitur = joblib.load(FEATURE_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal memuat model: {e}")

def prediksi_jurusan(jawaban_user):
    if not isinstance(jawaban_user, list) or len(jawaban_user) != len(fitur):
        raise ValueError(f"Input harus berupa list berisi {len(fitur)} angka (1–5)")
    input_array = np.array([jawaban_user])
    hasil = model.predict(input_array)
    return hasil[0]
