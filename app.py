import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained deepfake detection model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("real_fake.h5")
model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  
    img = np.array(img) / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

# Streamlit UI
st.title("Deepfake Image Detection")
st.write("Upload an image to check if it's Real or Fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(image)

    # Make prediction
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    # Display result
    if confidence > 0.5:
        st.error(f"This image is FAKE! (Confidence: {confidence:.2%})")
    else:
        st.success(f"This image is REAL! (Confidence: {100 - confidence:.2%})")
