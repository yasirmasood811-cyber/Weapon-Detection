import streamlit as st
import requests
from PIL import Image

url = "http://127.0.0.1:8000/predict"

st.title("YOLO Object Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("Original Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    files = {"file": (uploaded_file.name, uploaded_file.getbuffer())}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()

        st.subheader("Detections")
        for det in result.get("detections", []):
            st.write(f"Class: {det['class']}, Confidence: {det['confidence']:.2f}, BBox: {det['bbox']}")

        st.subheader("Annotated Image")
        with open("output.jpg", "rb") as f:
            img = Image.open(f)
            st.image(img, caption="YOLO Prediction", use_container_width=True)
    else:
        st.error(f"Error: {response.text}")
