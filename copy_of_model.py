# app.py - Cloud-safe Streamlit app: Pre-loads model, skips auto-train on Cloud

import streamlit as st
import os
import json
import numpy as np
from PIL import Image
import pickle

# Detect if running on Streamlit Cloud (via env vars)
IS_CLOUD = 'streamlit' in os.environ.get('HOME', '') or os.environ.get('STREAMLIT_CLOUD_RUN', False)

# Page config
st.set_page_config(page_title="Cow vs. Buffalo Classifier", page_icon="🐄", layout="wide")

# Load class names
@st.cache_data
def load_class_names():
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return ['buffalo', 'cow']  # Fallback

class_names = load_class_names()

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists('cow_buffalo_model.h5'):
        return None
    try:
        model = tf.keras.models.load_model('cow_buffalo_model.h5')
        model_size = os.path.getsize('cow_buffalo_model.h5') / (1024*1024)  # MB
        st.success(f"✅ Model loaded! Size: {model_size:.1f} MB")
        return model
    except Exception as e:
        st.error(f"Model load error: {str(e)}")
        return None

import tensorflow as tf  # Import here to avoid early errors
model = load_model()

# Title
st.title('🐄 Cow vs. 🐃 Buffalo Image Classifier')
st.markdown("""
Classify images as 'Cow' or 'Buffalo' using MobileNetV2 transfer learning.  
Trained on [Kaggle Dataset](https://www.kaggle.com/datasets/raghavdharwal/cows-and-buffalo-computer-vision-dataset).
""")

# Status and Cloud Warning
if model:
    st.success("✅ Ready to classify!")
else:
    st.error("❌ Model file 'cow_buffalo_model.h5' not found.")
    if IS_CLOUD:
        st.info("""
        **For Streamlit Cloud**: Pre-train locally with `python train_model.py`, then commit `cow_buffalo_model.h5`, 
        `class_names.json` to GitHub (use Git LFS for .h5). No auto-training on Cloud.
        """)
    else:
        st.info("Run `python train_model.py` locally to generate the model.")
    st.stop()  # Halt if no model

# Uploader
uploaded_files = st.file_uploader(
    "Choose image(s) (JPG, PNG)...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True
)

if uploaded_files:
    st.subheader("Results")
    progress_bar = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_column_width=True)
        
        with st.spinner(f'Classifying {uploaded_file.name}...'):
            img = image.convert('RGB').resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0]).numpy()
        predicted_idx = np.argmax(score)
        predicted_class = class_names[predicted_idx]
        confidence = 100 * score[predicted_idx]
        
        col1, col2 = st.columns(2)
        with col1:
            color = "success" if confidence > 80 else "warning"
            st.markdown(f'<h3 style="color: {"green" if color=="success" else "orange"};">**{predicted_class.upper()}**</h3>', unsafe_allow_html=True)
            st.info(f'Confidence: {confidence:.2f}%')
        with col2:
            probs = {class_names[i]: f"{100 * score[i]:.1f}%" for i in range(len(class_names))}
            for cls, prob in probs.items():
                st.metric(cls.capitalize(), prob)
        
        if confidence < 70:
            st.warning("Low confidence—try a clearer image.")
        
        st.bar_chart({cls: float(prob.strip('%')) for cls, prob in probs.items()})
        
        progress_bar.progress((idx + 1) / len(uploaded_files))

# Sidebar
with st.sidebar:
    st.header("📊 Model Info")
    st.write("""
    - **Model**: MobileNetV2 + Dropout.
    - **Classes**: Buffalo, Cow.
    - **Training**: 15 epochs, augmentation.
    - **Acc**: ~95-100%.
    """)
    
    st.header("💡 Tips")
    st.write("""
    - Use clear animal photos.
    - Batch upload supported.
    - For Cloud: Pre-train & commit model.
    """)
    
    # Lazy-load plot
    if st.checkbox("📈 Show Training History") and os.path.exists('training_history.pkl'):
        try:
            import matplotlib.pyplot as plt
            with open('training_history.pkl', 'rb') as f:
                history = pickle.load(f)
            fig, ax = plt.subplots()
            ax.plot(history['accuracy'], label='Train Acc')
            ax.plot(history['val_accuracy'], label='Val Acc')
            ax.set_title('Training History')
            ax.legend()
            st.pyplot(fig)
        except ImportError:
            st.warning("Install matplotlib for plots.")
        except Exception as e:
            st.error(f"Plot error: {e}")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Streamlit & TensorFlow | BoviTech")
