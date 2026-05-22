#For Streamlit
import streamlit as st
import torch
import re
import string
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Tokopedia Sentiment Analysis",
    page_icon="🛒",
    layout="wide"
)

# ==========================================
# 2. FUNGSI PREPROCESSING (SAMA DENGAN MODEL 4.1)
# ==========================================
def clean_text_refined(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'&\w+;', '', text)
    # ANGKA DIPERTAHANKAN (Sangat penting buat konteks Bintang 1-5)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 3. LOAD MODEL & TOKENIZER (CACHED)
# ==========================================
@st.cache_resource
def load_model():
    # Pastikan folder model_final ada di direktori yang sama
    model_path = "./model_final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# Load modelnya
tokenizer, model = load_model()

# ==========================================
# 4. SIDEBAR - PROJECT INFO & COMPARISON
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/af/Tokopedia_logo.svg", width=150)
    st.title("Project Info")
    st.markdown("""
    **Dataset:** [Tokopedia Reviews 2025](https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025)
    **Architecture:** IndoBERT Base P1
    **Optimization:** Focal Loss + Dampened Alpha
    """)
    
    st.divider()
    st.subheader("Model Comparison (Macro F1)")
    # Tabel perbandingan buat pamer progres ke dosen
    comparison_data = {
        "Model": ["Model 1 (Baseline)", "Model 2 (SMOTE)", "Model 3 (Weighted)", "Model 4.1 (Focal)"],
        "Macro F1": ["0.38", "0.50", "0.52", "0.63"]
    }
    st.table(pd.DataFrame(comparison_data))

# ==========================================
# 5. MAIN UI - PREDICTION
# ==========================================
st.title("🛍️ Tokopedia Sentiment Analyzer")
st.write("Masukkan review pelanggan di bawah untuk menganalisis sentimen secara otomatis.")

# Input area
user_input = st.text_area(
    "Review Text:", 
    placeholder="Contoh: Barang rusak pas nyampe, kecewa banget bintang 1...",
    height=150
)

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        # Preprocessing
        cleaned = clean_text_refined(user_input)
        
        # Inference
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = torch.max(probs).item()

        # Mapping Label (0: Negatif, 1: Netral, 2: Positif)
        label_map = {
            0: {"label": "NEGATIVE", "color": "red", "icon": "😡"},
            1: {"label": "NEUTRAL", "color": "orange", "icon": "😐"},
            2: {"label": "POSITIVE", "color": "green", "icon": "😊"}
        }
        
        res = label_map[prediction]

        # Tampilkan Hasil
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### Result: :{res['color']}[{res['label']} {res['icon']}]")
            st.write(f"**Processed Text:** *\"{cleaned}\"*")
            
        with col2:
            st.metric("Confidence Score", f"{confidence:.2%}")
            st.progress(confidence)

        # Analysis Alert
        if prediction == 0:
            st.error("Saran: Segera hubungi customer untuk komplain ini.")
        elif prediction == 1:
            st.warning("Catatan: Review ini bersifat informatif namun standar.")
        else:
            st.success("Bagus! Pelanggan puas dengan produk/layanan ini.")
            
    else:
        st.info("Tulis review-nya dulu dong Amang.")

# Footer
st.divider()
st.caption("Developed for NLP Project © 2026")