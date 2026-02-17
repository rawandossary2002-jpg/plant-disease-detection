import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224

# Simple treatment info (you can expand this list)
disease_info = {
    "Tomato - Late blight": "Remove infected leaves, avoid overhead watering, and use an appropriate fungicide if needed.",
    "Tomato - Early blight": "Remove infected leaves, improve airflow, rotate crops, and consider a fungicide spray.",
    "Tomato - Leaf Mold": "Reduce humidity, increase ventilation, remove infected leaves, and use fungicide if severe.",
    "Tomato - Septoria leaf spot": "Remove infected leaves, avoid watering leaves, and apply fungicide if required.",
    "Tomato - Bacterial spot": "Remove infected parts, avoid overhead watering, and use copper-based sprays if recommended.",
    "Apple - Apple scab": "Remove fallen leaves, prune for airflow, and apply fungicide during infection periods.",
    "Grape - Esca (Black Measles)": "Prune infected wood, sanitize tools, and maintain good vineyard hygiene.",
    "Strawberry - Leaf scorch": "Remove infected leaves, improve airflow, and avoid overhead irrigation.",
    "Pepper, bell - Bacterial spot": "Use disease-free seeds, avoid overhead irrigation, and apply copper sprays if advised.",
    "Corn - healthy": "No disease detected. Maintain good watering and nutrition.",
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

@st.cache_data
def load_classes():
    with open("classes.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def format_label(label: str) -> str:
    # Example: Tomato___Late_blight -> Tomato - Late blight
    return label.replace("___", " - ").replace("_", " ")

def prep_image(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)
    return img, x

# ---------- UI ----------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect the disease (PlantVillage classes).")

model = load_model()
classes = load_classes()

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    pil_img = Image.open(file)
    show_img, x = prep_image(pil_img)

    st.image(show_img, caption="Uploaded Image", use_container_width=True)

    preds = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    label = format_label(classes[top_idx])
    confidence = float(preds[top_idx]) * 100

    st.markdown("## 🌿 Prediction Result ✅")
    st.success(f"🌱 Detected: {label}")
    st.info(f"📊 Confidence: {confidence:.2f}%")

    # Treatment / advice
    st.subheader("Suggested Treatment 💊")
    if label in disease_info:
        st.write(disease_info[label])
    else:
        st.write("Treatment information is not available for this class yet.")

    # Top 5 predictions
    st.caption("Top 5 Predictions:")
    top5 = np.argsort(preds)[-5:][::-1]
    for i in top5:
        st.write(f"- {format_label(classes[int(i)])}: {float(preds[int(i)])*100:.2f}%")
