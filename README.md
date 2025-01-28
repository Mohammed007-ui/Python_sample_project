# Python_sample_project
saoftware_code
import cv2
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Load the YOLOv8 model (replace with your trained model path if needed)
model = YOLO("yolov8n.pt")  # You may need a custom-trained model for better accuracy

st.title("Surgical Tool Recognition")
st.write("Detects tools and displays their names and measurements in real-time.")

# OpenCV Video Capture
video_source = st.sidebar.selectbox("Select Video Source", [0, "Upload Video"])

if video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_video_path)
    else:
        cap = None
else:
    cap = cv2.VideoCapture(0)

stframe = st.empty()

# Tool names and reference measurements
TOOL_NAMES = ["Tool1", "Tool2", "Tool3", "Tool4"]
TOOL_MEASUREMENTS = {"Tool1": "10cm", "Tool2": "15cm", "Tool3": "12cm", "Tool4": "20cm"}

while cap is not None and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO model on the frame
    results = model(frame)
    
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            cls = int(cls)
            if cls < len(TOOL_NAMES):
                label = f"{TOOL_NAMES[cls]} ({TOOL_MEASUREMENTS[TOOL_NAMES[cls]]})"
            else:
                label = "Unknown Tool"
                
            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame in Streamlit
    stframe.image(frame, channels="BGR", use_column_width=True)

cap.release()
st.write("Stopped.")

