import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("emotion_recognition_model.keras")

# Define class labels
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Streamlit UI
st.title("Face Emotion Recognition App")
st.write("Upload an image and the model will predict the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to proper format
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (224, 224))  # Resize to match model input
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch

    # Prediction
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]

    st.write(f"Predicted Emotion: **{predicted_label}**")

