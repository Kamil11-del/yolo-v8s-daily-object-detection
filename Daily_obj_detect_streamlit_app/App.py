import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLOv8 model
model_path = 'obj_detect_model_weights.pt'  # Update with your trained model path
model = YOLO(model_path)

# Streamlit app title
st.title("YOLOv8 Daily Object Detection")

# Sidebar for uploading image
st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Helper function to load and preprocess image
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

# Main section - object detection
if uploaded_file is not None:
    # Load the uploaded image
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform object detection
    results = model.predict(image)
    
    # Extract the bounding boxes, class names, and confidence scores
    annotated_image = results[0].plot()  # Plot detections on image
    boxes = results[0].boxes.xyxy  # Bounding boxes
    labels = results[0].names  # Class names
    
    # Show detections on Streamlit app
    st.image(annotated_image, caption="Detected Objects", use_column_width=True)

    # Display detailed information about the detected objects
    st.write("Detected Objects:")
    for i, box in enumerate(boxes):
        st.write(f"Object {i+1}:")
        st.write(f"Class: {labels[i]}")
        st.write(f"Bounding Box: {box}")
        st.write(f"Confidence: {results[0].boxes.conf[i]}")
else:
    st.write("Upload an image to start object detection.")

# Optional: Add footer or additional Streamlit components
st.sidebar.write("Developed using YOLOv8 and Streamlit")
