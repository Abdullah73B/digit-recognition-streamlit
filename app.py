
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
import pickle
import tensorflow as tf

st.set_page_config(page_title="Handwritten Recognition", page_icon="🔢", layout="wide")

st.title("🔢 Handwritten Digit/Character Recognition")
st.write("Upload a **photo** of a handwritten digit or character (real-life photo/scan). App will preprocess and predict using the available model.")


@st.cache_resource
def load_models():
    models = {}
    
    if os.path.exists("model.pkl"):
        try:
            with open("model.pkl", "rb") as f:
                models['sklearn'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load model.pkl: {e}")
    
    if os.path.exists("D:\Final year Project\Digit Recgonition Deployment\model.h5"):
        try:
            models['keras'] = tf.keras.models.load_model("model.h5")
        except Exception as e:
            st.warning(f"Failed to load model.h5: {e}")
    return models

models = load_models()

if not models:
    st.error("No model found. Place either model.pkl (scikit-learn) or model.h5 (Keras) in the same folder. "
             "You can run the training scripts provided to create them.")
    st.stop()

st.sidebar.write("Detected models:")
for k in models.keys():
    st.sidebar.write(f"- {k}")

uploaded = st.file_uploader("Upload an image (photo of handwritten digit/character)", type=["png", "jpg", "jpeg"])

def preprocess_for_prediction(pil_image, target_size=(28,28), show_steps=False):
    """
    Preprocesses arbitrary photo to a normalized 28x28 grayscale image suitable for prediction:
    - Convert to grayscale
    - Invert/threshold optionally
    - Resize while maintaining aspect ratio and center
    - Normalize to [0,1]
    Returns numpy array of shape (28,28) float32
    """
    
    img = pil_image.convert("RGB")
    img_np = np.array(img)

 
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

   
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
       
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)
        roi = gray[y1:y2, x1:x2]
    else:
        
        roi = gray

    h, w = roi.shape
    
    if h > w:
        scale = (target_size[0] - 4) / h
        nh = target_size[0] - 4
        nw = int(w * scale)
    else:
        scale = (target_size[1] - 4) / w
        nw = target_size[1] - 4
        nh = int(h * scale)
    resized = cv2.resize(roi, (nw, nh))
   
    canvas = np.zeros(target_size, dtype=np.uint8)
    x_off = (target_size[1] - nw) // 2
    y_off = (target_size[0] - nh) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

    
    img_out = canvas.astype("float32") / 255.0
    return img_out

if uploaded:
    pil_image = Image.open(uploaded)
    st.subheader("Uploaded Image")
    st.image(pil_image, use_column_width=True)

    
    processed = preprocess_for_prediction(pil_image, target_size=(28,28))
    st.subheader("Preprocessed 28x28 (what the model sees)")
    
    st.image((processed * 255).astype("uint8"), clamp=True, width=150)

   
    predict_btn = st.button("Predict")

    if predict_btn:
        if 'keras' in models:
            model = models['keras']
            x = processed.reshape(1,28,28,1)
            preds = model.predict(x, verbose=0)[0]
            pred = int(np.argmax(preds))
            conf = float(np.max(preds))
            st.success(f"Keras model prediction: {pred}  (confidence: {conf*100:.2f}%)")
            
            st.write("Probabilities:")
            probs = {str(i): float(preds[i]) for i in range(len(preds))}
            st.write(probs)
        elif 'sklearn' in models:
            model = models['sklearn']
            x = processed.flatten().reshape(1, -1)
            pred = model.predict(x)[0]
            conf = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
                conf = float(np.max(probs))
                st.success(f"Sklearn model prediction: {pred}  (confidence: {conf*100:.2f}%)")
                st.write({str(i): float(probs[i]) for i in range(len(probs))})
            else:
                st.success(f"Sklearn model prediction: {pred}")
        else:
            st.error("No usable model loaded.")
