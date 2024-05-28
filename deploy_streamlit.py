<<<<<<< HEAD
<<<<<<< HEAD
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('model_cnn.h5')

# Function to load and preprocess image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    return img_array

# Streamlit UI
st.title("Klasifikasi Penyakit Ikan Dengan CNN")
st.write("Tambahkan Gambar Untuk Klasifikasi")

uploaded_file = st.file_uploader("Pilih Gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption='Gambar Berhasil Diunggah', use_column_width=True)

    # Button to start the classification
    if st.button("Deteksi"):
        # Load and preprocess the image
        img_array = load_and_preprocess_image("temp_image.jpg")

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        # Define class labels
        labels = ['Aeromoniasis', 'Bacterial Gill Disease', 'Parasit', 'Saprolegniasis', 'Tail and Fin Rot']

        # Display the prediction result
        st.write("Hasil Klasifikasi: ")
=======
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('model_cnn.h5')

# Function to load and preprocess image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    return img_array

# Streamlit UI
st.title("Klasifikasi Penyakit Ikan Dengan CNN")
st.write("Tambahkan Gambar Untuk Klasifikasi")

uploaded_file = st.file_uploader("Pilih Gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption='Gambar Berhasil Diunggah', use_column_width=True)

    # Button to start the classification
    if st.button("Deteksi"):
        # Load and preprocess the image
        img_array = load_and_preprocess_image("temp_image.jpg")

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        # Define class labels
        labels = ['Aeromoniasis', 'Bacterial Gill Disease', 'Parasit', 'Saprolegniasis', 'Tail and Fin Rot']

        # Display the prediction result
        st.write("Hasil Klasifikasi: ")
>>>>>>> 26c4409aaa43f0338eaf46d334ca3da5e5074176
        st.success(f"{labels[predicted_class[0]]}")
=======
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Function to load and preprocess image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    return img_array

# Streamlit UI
st.title("Klasifikasi Penyakit Ikan Dengan CNN")
st.write("Tambahkan Gambar Untuk Klasifikasi")

uploaded_file = st.file_uploader("Pilih Gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption='Gambar Berhasil Diunggah', use_column_width=True)

    # Button to start the classification
    if st.button("Deteksi"):
        # Load and preprocess the image
        img_array = load_and_preprocess_image("temp_image.jpg")

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        # Define class labels
        labels = ['Aeromoniasis', 'Bacterial Gill Disease', 'Parasit', 'Saprolegniasis', 'Tail and Fin Rot']

        # Display the prediction result
        st.write("Hasil Klasifikasi: ")
        st.success(f"{labels[predicted_class[0]]}")
>>>>>>> 4fa62d64fe9b289b2c7ed0e4bea952efda46da84
