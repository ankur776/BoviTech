# app.py - Streamlit app for Cow vs. Buffalo classification
# Loads the trained H5 model and classifies uploaded images

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Page config for better layout
st.set_page_config(
    page_title="Cow vs. Buffalo Classifier",
    page_icon="🐄",
    layout="wide"
)

# Load class names (cached, with fallback)
@st.cache_data
def load_class_names():
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("class_names.json not found. Run train_model.py first to generate it.")
        return ['buffalo', 'cow']  # Fallback (alphabetical)

class_names = load_class_names()

# Load the trained model (cached for efficiency)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('cow_buffalo_model.h5')
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("cow_buffalo_model.h5 not found. Run train_model.py first to train and save the model.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Check TensorFlow version compatibility.")
        return None

model = load_model()

# Main title and description
st.title('🐄 Cow vs. 🐃 Buffalo Image Classifier')
st.markdown("""
Upload an image below to classify it as 'Cow' or 'Buffalo' using a MobileNetV2 transfer learning model.  
The model was trained on the [Kaggle Cows and Buffalo Computer Vision Dataset](https://www.kaggle.com/datasets/raghavdharwal/cows-and-buffalo-computer-vision-dataset)  
with data augmentation for better generalization.
""")

# File uploader (supports multiple files for batch testing)
uploaded_files = st.file_uploader(
    "Choose image file(s) (JPG, JPEG, or PNG)...", 
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files and model is not None:
    st.subheader("Classification Results")
    for uploaded_file in uploaded_files:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption=f'Uploaded: {uploaded_file.name}', use_column_width=True)
        
        # Preprocess (match training exactly: RGB, resize, normalize)
        with st.spinner(f'Classifying {uploaded_file.name}...'):
            img = image.convert('RGB').resize((224, 224))  # Ensure RGB
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
            img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Predict
        try:
            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0]).numpy()  # Probabilities
            predicted_idx = np.argmax(score)
            predicted_class = class_names[predicted_idx]
            confidence = 100 * score[predicted_idx]
            
            # Results layout
            col1, col2 = st.columns([2, 1])
            with col1:
                if confidence > 80:
                    st.success(f'**Predicted: {predicted_class.upper()}**')
                else:
                    st.warning(f'**Predicted: {predicted_class.upper()}** (Low confidence)')
                st.info(f'Confidence: {confidence:.2f}%')
            with col2:
                probs = {class_names[i]: f"{100 * score[i]:.1f}%" for i in range(len(class_names))}
                for cls, prob in probs.items():
                    st.metric(cls.capitalize(), prob)
            
            # Confidence threshold warning
            if confidence < 70:
                st.warning("Low confidence prediction. Try a clearer image of a cow or buffalo.")
            
            # Bar chart for visual
            st.bar_chart({cls: float(prob.strip('%')) for cls, prob in probs.items()})
            
        except Exception as e:
            st.error(f"Prediction failed for {uploaded_file.name}: {str(e)}")
elif uploaded_files and model is None:
    st.warning("Files uploaded, but model not loaded. Train the model first with train_model.py.")

# Sidebar with model info and tips
with st.sidebar:
    st.header("📊 Model Details")
    st.write("""
    - **Architecture**: MobileNetV2 (ImageNet pre-trained, frozen base) + Dense head with dropout.
    - **Classes**: Buffalo and Cow (auto-detected from dataset folders).
    - **Training**: 15 epochs, Adam optimizer, categorical cross-entropy.
    - **Data**: ~1000 images from Kaggle, with augmentation (rotation, flips, shifts).
    - **Performance**: Validation accuracy ~95-100% (dropout added to reduce overfitting).
    """)
    
    st.header("💡 Tips for Best Results")
    st.write("""
    - Use clear, front-facing photos of cows or buffaloes.
    - Avoid blurry, dark, or non-animal images (may give low confidence).
    - Model excels on dataset-like images (e.g., farm animals).
    - Batch upload: Select multiple files to classify at once.
    - Deploy: Push to GitHub and use Streamlit Cloud for sharing.
    """)
    
    # Optional: Training history plot (uncomment if you save history in train_model.py)
    # if st.checkbox("Show Training History"):
    #     try:
    #         import pickle
    #         import matplotlib.pyplot as plt
    #         with open('training_history.pkl', 'rb') as f:
    #             history = pickle.load(f)
    #         fig, ax = plt.subplots()
    #         ax.plot(history['accuracy'], label='Training Accuracy')
    #         ax.plot(history['val_accuracy'], label='Validation Accuracy')
    #         ax.set_xlabel('Epoch')
    #         ax.set_ylabel('Accuracy')
    #         ax.legend()
    #         st.pyplot(fig)
    #     except FileNotFoundError:
    #         st.warning("Training history file not found.")

# Footer
st.markdown("---")
st.caption("🤖 Built with Streamlit & TensorFlow | Project: BoviTech | Questions? Check the sidebar.")
