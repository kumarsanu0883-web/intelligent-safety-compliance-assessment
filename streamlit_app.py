import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🦺 Intelligent Safety Compliance Assessment")

model = YOLO("weights/best.pt")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    results = model(img)

    output = results[0].plot()

    st.image(
        output,
        caption="Safety Detection Result",
        use_container_width=True
    )