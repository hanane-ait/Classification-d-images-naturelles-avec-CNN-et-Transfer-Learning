
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Classification",
    page_icon="🍎"
)

st.title("Classification Application")
st.write(
    "Upload an image of a fruit to identify whether it is "
    "a banana, an orange or an apple."
)

# Chemin du modèle CNN
model_path = Path(__file__).parent / "model.h5"

st.write("Model path:", str(model_path))

# Chargement du modèle
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()

try:
    model = load_model(str(model_path))
    st.success("CNN model loaded successfully!")

except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.exception(e)
    st.stop()

# Classes du modèle
classes = ["Apple", "Banana", "Orange"]

# Importation de l'image
uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        # Lecture et affichage de l'image
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=300
        )

        # Redimensionnement selon l'entrée du modèle
        height, width = model.input_shape[1:3]
        img = image.resize((width, height))

        # Conversion en tableau NumPy
        # Aucune normalisation ici
        img_array = np.array(img, dtype=np.float32)

        # Ajout de la dimension du batch
        img_array = np.expand_dims(img_array, axis=0)

        # Prédiction
        predictions = model.predict(img_array, verbose=0)
        scores = predictions[0]

        # Vérification du nombre de classes
        if len(scores) != len(classes):
            st.error(
                f"The model returns {len(scores)} classes, "
                f"but {len(classes)} labels are configured."
            )
            st.stop()

        # Résultat principal
        index = int(np.argmax(scores))

        st.subheader("Prediction Results")
        st.success(f"Predicted fruit: {classes[index]}")
        st.write(f"Confidence: {scores[index] * 100:.2f}%")

        # Probabilités de chaque classe
        st.write("### Class predictions")

        for label, score in zip(classes, scores):
            st.write(f"**{label}:** {score * 100:.2f}%")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)

else:
    st.info("Please upload an image to classify.")
