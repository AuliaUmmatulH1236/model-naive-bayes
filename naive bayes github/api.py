from fastapi import FastAPI  # Jika pydantic terinstal normal gunakan: from pydantic import BaseModel
from pydantic import BaseModel
import joblib
from typing import List
from preprocessing import preprocess_text, keyword_features_smart

# Load model bundle baru yang sudah menyimpan tfidf_stage1 dan tfidf_stage2
model_bundle = joblib.load ("model_nbm+smote_tunning_sentimen2.pkl")

tfidf_stage1 = model_bundle["tfidf_stage1"]
tfidf_stage2 = model_bundle["tfidf_stage2"]
model_aspek = model_bundle["model_aspek"]
model_sentimen = model_bundle["model_sentimen"]
aspek_list = model_bundle["aspek_list"]

# Label disesuaikan dengan Istilah Baru (Tingkat Kepuasan) di Notebook Skripsi-mu
label_map = {0: "CUKUP PUAS", 1: "SANGAT PUAS", 2: "TIDAK PUAS"}

app = FastAPI()

# ======================
# INPUT SCHEMA
# ======================
class InputText(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

# ======================
# CORE FUNCTION (LOGIKA SINKRON DENGAN NOTEBOOK TRAINING)
# ======================
def process_text(text):
    # 1. Jalankan preprocessing dasar
    clean = preprocess_text(text)
    
    # 2. Jalankan fungsi asimilasi teks pintar
    text_asimilasi, tag_aspek = keyword_features_smart(clean)
    
    # 3. Bentuk teks input untuk Stage 1 & Stage 2 secara terpisah
    final_s1 = clean + " " + tag_aspek
    
    # 4. Transformasikan teks ke matriks TF-IDF independen masing-masing stage
    vec_s1 = tfidf_stage1.transform([final_s1])
    vec_s2 = tfidf_stage2.transform([text_asimilasi])

    temp_results = []

    # 5. Iterasi Prediksi per Aspek
    for asp in aspek_list:
        # Cek probabilitas Stage 1 (Deteksi Aspek)
        prob = model_aspek[asp].predict_proba(vec_s1)[0][1]

        if prob > 0.75:
            if asp in model_sentimen:
                # Cek prediksi kelas Stage 2 (Tingkat Kepuasan) menggunakan vec_s2
                sent_code = model_sentimen[asp].predict(vec_s2)[0]

                temp_results.append({
                    "aspek": asp.upper(),
                    "kepuasan": label_map[sent_code]
                })

    # ====================================================
    # FILTER ASPEK UMUM (Bila ada Aspek Utama, hapus 'umum')
    # ====================================================
    aspek_utama = {"makanan", "harga", "layanan", "lingkungan"}
    detected = {x["aspek"].lower() for x in temp_results}

    if len(detected & aspek_utama) > 0:
        temp_results = [x for x in temp_results if x["aspek"].lower() != "umum"]
        
    # Jika tidak ada aspek apapun yang lolos threshold, default ke UMUM
    if not temp_results:
        if "umum" in model_sentimen:
            sent_code = model_sentimen["umum"].predict(vec_s2)[0]
            sentimen_default = label_map[sent_code]
        else:
            sentimen_default = "CUKUP PUAS"
        temp_results.append({
            "aspek": "UMUM",
            "kepuasan": sentimen_default
        })

    return {
        "input": text,
        "clean": clean,
        "hasil": temp_results
    }

# ======================
# ENDPOINTS FastAPI
# ======================
@app.post("/predict")
def predict(data: InputText):
    return process_text(data.text)

@app.post("/predict_batch")
def predict_batch(data: BatchInput):
    results_all = []
    for text in data.texts:
        results_all.append(process_text(text))
    return results_all