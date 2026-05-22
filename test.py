import torch
import os
import re
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# PREPROCESSING (WAJIB KONSISTEN)
# ============================================================
def clean_text_refined(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'&\w+;', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================
# DAFTAR TEKS UJI (BERBAGAI SKENARIO)
# ============================================================
sample_reviews = [
    "Barang bagus banget pengiriman cepat, puas belanja di sini!", # Positif Jelas
    "Produk jelek parah, pas dibuka hancur. Jangan beli di sini.", # Negatif Jelas
    "Lumayan lah dengan harga segini, standar kualitasnya.",      # Netral Biasa
    "Bintang 1 buat toko ini, kapok banget belanja disini.",       # Angka (Sinyal Negatif)
    "Kualitas produk sih oke, tapi kurirnya lama banget sampai.",   # Campuran (Netral/Arah Negatif)
    "Bagus banget ya, dipake sekali langsung jebol wkwk"          # Sarkasme (Sulit)
]

def run_batch_prediction(model_path, model_name):
    print(f"\n" + "="*70)
    print(f"🔮 MEMULAI PREDIKSI BATCH: {model_name}")
    print(f"📂 Path: {model_path}")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Folder {model_path} tidak ditemukan!")
        return

    try:
        # Load Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        label_map = {0: "NEGATIF 😡", 1: "NETRAL 😐", 2: "POSITIF 😊"}

        for idx, text in enumerate(sample_reviews, 1):
            cleaned = clean_text_refined(text)
            inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = torch.max(probs).item()
            
            print(f"\n[{idx}] Teks Asli : \"{text}\"")
            print(f"    Hasil      : {label_map.get(pred_class, 'Unknown')} ({confidence:.2%})")
            print(f"    Probs      : [Neg: {probs[0][0]:.3f} | Net: {probs[0][1]:.3f} | Pos: {probs[0][2]:.3f}]")
            
    except Exception as e:
        print(f"❌ Gagal memproses! Error: {e}")

if __name__ == "__main__":
    run_batch_prediction("./model_results/03_Model", "Model 3 (Weighted CE)")
    run_batch_prediction("./model_results/04_Model", "Model 4 (Focal Loss)")