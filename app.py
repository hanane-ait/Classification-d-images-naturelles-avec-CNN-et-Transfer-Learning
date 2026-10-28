
import json
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Natural Images Classification",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Natural Images Classification")
st.write(
    "Importez une image pour identifier sa classe "
    "à l'aide du modèle CNN sélectionné dans votre notebook."
)

# Chemins des fichiers
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.h5"
CLASSES_PATH = BASE_DIR / "class_names.json"

# Vérifier la présence des fichiers
if not MODEL_PATH.exists():
    st.error(f"Modèle introuvable : {MODEL_PATH}")
    st.stop()

if not CLASSES_PATH.exists():
    st.error(f"Fichier des classes introuvable : {CLASSES_PATH}")
    st.stop()

# Charger le modèle
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

# Charger les classes
@st.cache_data
def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

try:
    model = load_model(str(MODEL_PATH))
    classes = load_classes(str(CLASSES_PATH))

    st.success("Modèle chargé avec succès !")

except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    st.stop()

# Vérifier la compatibilité du modèle
if model.output_shape[-1] != len(classes):
    st.error(
        f"Le modèle produit {model.output_shape[-1]} sorties, "
        f"mais le fichier contient {len(classes)} classes."
    )
    st.stop()

# Importer une image
uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Image importée",
            width=350
        )

        # Récupérer la taille attendue par le modèle
        height, width = model.input_shape[1:3]

        if height is None or width is None:
            height, width = 150, 150

        # Préparer l'image sans normalisation supplémentaire :
        # le prétraitement du modèle sauvegardé est conservé.
        resized_image = image.resize((width, height))

        img_array = np.asarray(
            resized_image,
            dtype=np.float32
        )

        img_array = np.expand_dims(img_array, axis=0)

        # Prédiction
        with st.spinner("Classification en cours..."):
            predictions = model.predict(img_array, verbose=0)[0]

        if len(predictions) != len(classes):
            st.error("Le nombre de prédictions est incorrect.")
            st.stop()

        predicted_index = int(np.argmax(predictions))
        predicted_class = classes[predicted_index]
        confidence = float(predictions[predicted_index])

        st.subheader("Résultat de la classification")
        st.success(f"Classe prédite : {predicted_class}")
        st.metric(
            "Score du modèle",
            f"{confidence * 100:.2f}%"
        )

        # Afficher les scores de toutes les classes
        st.subheader("Scores par classe")

        scores = sorted(
            zip(classes, predictions),
            key=lambda item: item[1],
            reverse=True
        )

        for label, score in scores:
            st.write(f"**{label}** — {score * 100:.2f}%")
            st.progress(
                float(np.clip(score, 0, 1))
            )

    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")

else:
    st.info("Importez une image pour commencer.")
