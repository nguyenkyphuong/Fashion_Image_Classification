import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

CLASS_NAMES = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

@st.cache_resource
def get_model(path="fashion_mnist_model.keras"):
    return load_model(path)

def preprocess(img):
    img = ImageOps.grayscale(img).resize((28, 28))
    x = (np.array(img).astype("float32") / 255.0)[None, ..., None]
    return x

def predict(img):
    x = preprocess(img)
    proba = get_model().predict(x, verbose=0)[0]
    idx = int(np.argmax(proba))
    return idx, proba

st.set_page_config(page_title="Fashion-MNIST Classifier", page_icon="👗")
st.title("Fashion-MNIST Image Classification 👗👟")
st.caption("Upload a clothing image; the app resizes to 28×28 grayscale and predicts the category.")

file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
if file:
    img = Image.open(file).convert("RGB")
    col1, col2 = st.columns([1, 1.6])
    with col1:
        st.image(img, caption="Input", width=220)
    idx, proba = predict(img)
    with col2:
        st.markdown(f"### Predicted: **{CLASS_NAMES[idx]}**")
        st.markdown(f"Confidence: **{proba[idx]:.2%}**")
        st.write("Top-3 probabilities:")
        top3 = np.argsort(proba)[-3:][::-1]
        for i in top3:
            st.write(f"- {CLASS_NAMES[i]}: {proba[i]:.2%}")
