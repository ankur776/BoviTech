# app.py - Updated Streamlit app with integrated training trigger
# Auto-trains if model missing; supports batch upload and demo mode
# Matplotlib import moved inside conditional (no crash if not installed)

import streamlit as st
import tensorflow as tf
import subprocess
import sys
import os
import json
import numpy as np
from PIL import Image
import pickle

# Page config
st.set_page_config(
    page_title="Cow vs. Buffalo Classifier",
    page_icon="🐄",
    layout="wide"
)

# Function to run training script if model missing
@st.cache_resource
def ensure_model_trained():
    if not os.path.exists('cow_buffalo_model.h5'):
        st.sidebar.warning("Model not found. Training now...")
        try:
            # Run train_model.py as subprocess
            result = subprocess.run([sys.executable, 'train_model.py'], capture_output=True, text=True, timeout=1800)  # 30 min timeout
            if result.returncode != 0:
                st.error(f"Training failed: {result.stderr}")
                return None
            st.sidebar.success("Training completed!")
            st.rerun()  # Refresh to load model
        except subprocess.TimeoutExpired:
            st.error("Training timed out. Run 'python train_model.py' manually in terminal.")
            return None
        except FileNotFoundError:
            st.error("train_model.py not found. Ensure both .py files are in the same folder.")
            return None
    return True

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
    try:
        model = tf.keras.models.load_model('cow_buffalo_model.h5')
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Model load error: {str(e)}")
        return None

# Ensure trained and load
trained = ensure_model_trained()
model = load_model() if trained else None

# Title
st.title('🐄 Cow vs. 🐃 Buffalo Image Classifier')
st.markdown("""
Classify images as 'Cow' or 'Buffalo' using MobileNetV2 transfer learning.  
Trained on [Kaggle Dataset](https://www.kaggle.com/datasets/raghavdharwal/cows-and-buffalo-computer-vision-dataset) with augmentation.
""")

# Status
if model:
    st.success("✅ Model loaded and ready!")
else:
    st.warning("⚠️ Model not available. Use sidebar to train.")
    if st.button("Demo Mode (Dummy Prediction)"):
        st.info("Demo: This uses fallback classes. Upload a real image after training.")

# Uploader
uploaded_files = st.file_uploader(
    "Choose image(s) (JPG, PNG)...",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True
)

if uploaded_files and model:
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
elif uploaded_files and not model:
    st.warning("Upload ready, but train the model first via sidebar.")

# Sidebar
with st.sidebar:
    st.header("🔧 Controls")
    if not model:
        if st.button("🚀 Train Model Now", type="primary"):
            st.rerun()  # Triggers ensure_model_trained()
        st.info("Click above to auto-train (requires Kaggle setup). Or pre-train locally and commit model files.")
    
    st.header("📊 Model Info")
    st.write("""
    - **Model**: MobileNetV2 + Dropout (anti-overfitting).
    - **Classes**: Auto from dataset (buffalo, cow).
    - **Training**: 15 epochs, 80/20 split, augmentation.
    - **Acc**: ~95-100% validation.
    """)
    
    st.header("💡 Tips")
    st.write("""
    - Clear, lit images work best.
    - Batch: Upload multiple for quick tests.
    - Deploy: Pre-train model locally for Cloud (commit .h5 file).
    """)
    
    # Training history plot (lazy-load matplotlib)
    if st.checkbox("📈 Show Training History") and os.path.exists('training_history.pkl'):
        try:
            # Import matplotlib ONLY here (lazy-load: no crash if missing)
            import matplotlib.pyplot as plt
            with open('training_history.pkl', 'rb') as f:
                history = pickle.load(f)
            fig, ax = plt.subplots()
            ax.plot(history['accuracy'], label='Train Acc')
            ax.plot(history['val_accuracy'], label='Val Acc')
            ax.set_title('Training History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            st.pyplot(fig)
        except ImportError:
            st.warning("Matplotlib not installed. Install with 'pip install matplotlib' (or add to requirements.txt) to view plots.")
        except Exception as e:
            st.error(f"Plot error: {e}")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Streamlit & TensorFlow | BoviTech Project")
