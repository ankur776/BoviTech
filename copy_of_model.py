# app.py - Streamlit app for Cow vs. Buffalo classification
# Loads the trained H5 model and classifies uploaded images

import streamlit as st
from PIL import Image
import numpy as np
import json

# Page config
st.set_page_config(page_title="Cow vs. Buffalo Classifier", page_icon="🐄", layout="wide")

# Load class names (saved during training)
@st.cache_data
def load_class_names():
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("class_names.json not found. Run train_model.py first.")
        return ['cow', 'buffalo']  # Fallback

class_names = load_class_names()

# Load the trained model (cached for efficiency)
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('cow_buffalo_model.h5')
    except FileNotFoundError:
        st.error("cow_buffalo_model.h5 not found. Run train_model.py first.")
        return None

model = load_model()

# Title and description
st.title('🐄 Cow vs. 🐃 Buffalo Image Classifier')
st.write("""
Upload an image below to classify it as 'Cow' or 'Buffalo' using a MobileNetV2 transfer learning model.
The model was trained on the Kaggle 'Cows and Buffalo Computer Vision Dataset' with data augmentation.
""")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file (JPG, JPEG, or PNG)...", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None and model is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image (match training: resize to 224x224, normalize to [0,1])
    with st.spinner('Preprocessing and classifying...'):
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)
        img_array = img_array / 255.0  # Rescale (same as training)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0]).numpy()  # Softmax for probabilities
    predicted_idx = np.argmax(score)
    predicted_class = class_names[predicted_idx]
    confidence = 100 * score[predicted_idx]
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.success(f'**Predicted: {predicted_class.upper()}**')
        st.info(f'Confidence: {confidence:.2f}%')
    with col2:
        # Show probabilities for both classes
        probs = {class_names[i]: f"{100 * score[i]:.1f}%" for i in range(len(class_names))}
        st.metric("Class Probabilities", probs)
    
    # Optional: Show prediction bar chart
    st.subheader('Detailed Confidence Scores')
    st.bar_chart(probs)

elif uploaded_file is not None and model is None:
    st.warning("Model not loaded. Please train the model first with train_model.py.")

# Sidebar with info and tips
with st.sidebar:
    st.header("Model Details")
    st.write("""
    - **Architecture**: MobileNetV2 (pre-trained on ImageNet, frozen base layers) + custom head.
    - **Classes**: Cow and Buffalo (from dataset).
    - **Training**: 15 epochs, Adam optimizer, categorical cross-entropy loss.
    - **Preprocessing**: Images resized to 224x224, normalized to [0,1], with augmentation (rotation, flip, etc.).
    - **Performance**: Validation accuracy ~100% (monitor for overfitting; consider more data).
    """)
    st.header("Tips")
    st.write("""
    - Upload clear, well-lit images of cows or buffaloes.
    - Model works best on similar styles to the training dataset.
    - For production, deploy to Streamlit Cloud via GitHub.
    """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit and TensorFlow. Dataset: [Kaggle Cows and Buffalo CV](https://www.kaggle.com/datasets/raghavdharwal/cows-and-buffalo-computer-vision-dataset)")
