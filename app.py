import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import urllib.request

# 1. Sayfa Tasarımı
st.set_page_config(page_title="Melanoma Karar Destek Sistemi", layout="wide")
st.title("🩺 Akıllı Melanoma Karar Destek Sistemi")
st.write("Yapay zeka tabanlı segmentasyon modeli ile lezyon analizi prototipi.")

# 2. Google Drive'dan Model İndirme Fonksiyonu
@st.cache_resource
def load_model():
    model_filename = "best.pt"
    
    # Eğer model zaten indirilmişse tekrar indirme
    if not os.path.exists(model_filename):
        with st.spinner("Yapay zeka modeli buluttan indiriliyor, lütfen birkaç saniye bekleyin..."):
            # BURASI ÇOK ÖNEMLİ: Aşağıdaki tırnak içine Google Drive'dan kopyaladığın linki yapıştıracaksın!
            drive_url = "https://drive.google.com/file/d/1Z-qkgfw167vnTuAEDbXf3QB7HsmWGuXO/view?usp=sharing"
            
            # Google Drive linkini doğrudan indirme linkine çevirme işlemi
            if "file/d/" in drive_url:
                file_id = drive_url.split("file/d/")[1].split("/")[0]
                direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                urllib.request.urlretrieve(direct_download_url, model_filename)
            else:
                st.error("Lütfen geçerli bir Google Drive paylaşım linki girin!")
                return None

    return YOLO(model_filename)

try:
    model = load_model()
    if model:
        st.success("Yapay zeka modeli (best.pt) başarıyla yüklendi!")
except Exception as e:
    st.error(f"Model yüklenirken bir hata oluştu: {e}")

st.divider()

# 3. Panel Düzeni
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Test Görseli Yükleme")
    uploaded_file = st.file_uploader("Analiz edilecek lezyon fotoğrafını seçiniz...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Yüklenen Orijinal Görsel", use_container_width=True)
    
    with col2:
        st.subheader("🔍 Yapay Zeka Analiz Sonucu")
        
        with st.spinner("Model görseli analiz ediyor, lütfen bekleyin..."):
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            results = model(img_bgr)
            res = results[0]
            
            res_plotted = res.plot(labels=True, boxes=False)
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.image(res_plotted_rgb, caption="Yapay Zeka Segmentasyon Maskesi", use_container_width=True)
            
            if res.boxes is not None and len(res.boxes) > 0:
                confidence = res.boxes.conf[0].item()
                percentage = confidence * 100
                
                st.metric(label="Melanoma Olasılığı (Güven Skoru)", value=f"% {percentage:.2f}")
                
                if percentage > 70:
                    st.error("🚨 Yüksek Risk: En yakın zamanda bir dermatoloğa başvurulması önerilir.")
                elif percentage > 40:
                    st.warning("⚠️ Orta Risk: Takip edilmesi önerilir.")
                else:
                    st.info("✅ Düşük Risk: Model güven oranı düşük.")
            else:
                st.info("Sağlıklı / Düşük Risk: Model herhangi bir melanoma lezyonu tespit edemedi.")