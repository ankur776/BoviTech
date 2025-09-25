import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
# Removed joblib import
# import joblib

# Load the trained model in Keras format
model = tf.keras.models.load_model('cow_buffalo_model.keras')

# Load the class names from the JSON file
import json
with open("class_names.json", "r") as f:
    class_indices = json.load(f)
    # Reverse the dictionary to get class names from indices
    class_names = {v: k for k, v in class_indices.items()}


st.title('Cows and Buffalo Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) # Added more image types

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the same way as training data

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.write(f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score):.2f}% confidence.")
