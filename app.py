import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# Page config for better layout
st.set_page_config(
    page_title="Cow vs. Buffalo Classifier",
    page_icon="🐄",
    layout="wide"
)

# Title
st.title('🐄 Cow vs. 🐃 Buffalo Image Classifier')
st.markdown("Upload an image to classify it as 'Cow' or 'Buffalo' using a trained MobileNetV2 model.")

# Load class names (with fallback)
@st.cache_data
def load_class_names():
    if not os.path.exists('class_names.json'):
        st.error("❌ 'class_names.json' missing. Run train_model.py locally to generate it.")
        st.stop()
    try:
        with open("class_names.json", "r") as f:
            class_indices = json.load(f)  # e.g., {'buffalo': 0, 'cow': 1}
        # Convert to sorted list: [class for index 0, 1, ...]
        class_names = [None] * len(class_indices)
        for cls, idx in class_indices.items():
            class_names[idx] = cls
        return class_names
    except Exception as e:
        st.error(f"Error loading class names: {str(e)}")
        st.stop()

class_names = load_class_names()

# Load model (cached, with .h5 for compatibility)
@st.cache_resource
def load_model():
    model_path = 'cow_buffalo_model.h5'  # Changed to .h5 for better compatibility
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' missing. Train locally with train_model.py and commit the .h5 file.")
        st.stop()
    try:
        model = tf.keras.models.load_model(
            model_path,
            # Custom objects for MobileNetV2 (helps with deserialization)
            custom_objects={'MobileNetV2': tf.keras.applications.MobileNetV2}
        )
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        st.success(f"✅ Model loaded successfully! Size: {model_size:.1f} MB")
        return model
    except ValueError as e:
        if "input" in str(e).lower() or "shape" in str(e).lower():
            st.error("Model load failed due to input shape issue. Re-train with TF 2.13 and save as .h5.")
        else:
            st.error(f"Model load error (ValueError): {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected model load error: {str(e)}")
        st.stop()

model = load_model()

# File uploader (supports multiple files)
uploaded_files = st.file_uploader(
    "Choose image(s)...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("Classification Results")
    progress_bar = st.progress(0)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Read and display image
        image = Image.open(uploaded_file)
        st.image(image, caption=f'Uploaded: {uploaded_file.name}', use_column_width=True)
        
        # Preprocess (your logic, with RGB conversion for safety)
        img = image.convert('RGB').resize((224, 224))  # Ensure RGB
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
        img_array = img_array / 255.0  # Normalize
        
        # Make prediction (your logic)
        with st.spinner(f'Classifying {uploaded_file.name}...'):
            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])
            predicted_idx = np.argmax(score)
            predicted_class = class_names[predicted_idx]
            confidence = 100 * np.max(score)
        
        # Display result (your style, with color and metrics)
        col1, col2 = st.columns([2, 1])
        with col1:
            color = "🟢" if confidence > 80 else "🟡"  # Simple emoji for confidence
            st.markdown(f"**{color} Prediction: {predicted_class.upper()}**")
            st.write(f"Confidence: {confidence:.2f}%")
        with col2:
            for i, cls in enumerate(class_names):
                prob = 100 * score[i]
                st.metric(cls.capitalize(), f"{prob:.1f}%")
        
        if confidence < 70:
            st.warning("Low confidence—try a clearer image of the animal.")
        
        # Optional: Bar chart for probabilities
        st.bar_chart({cls: score[i] * 100 for i, cls in enumerate(class_names)})
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
        st.markdown("---")  # Separator between images

# Sidebar for info/tips
with st.sidebar:
    st.header("📋 About")
    st.write("""
    - **Model**: MobileNetV2 transfer learning.
    - **Input**: 224x224 RGB images.
    - **Dataset**: Kaggle cows/buffalo images.
    - **Tips**: Upload clear, well-lit photos. Supports batches!
    """)
    st.header("🔧 Setup")
    st.info("""
    If model/files missing:
    1. Run `python train_model.py` locally.
    2. Commit `.h5` and `.json` to GitHub.
    3. For Cloud: Use Git LFS for large .h5 files.
    """)

# Footer
st.markdown("---")
st.caption("🤖 Built with Streamlit & TensorFlow | Test with cow/buffalo photos!")
