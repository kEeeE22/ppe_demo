import streamlit as st
import cv2
from ultralytics import YOLO
from config import *
import time
import tempfile

st.sidebar.header('PPE Detection')
model_selectbox = st.sidebar.selectbox(
    'Choose model',
    ('rtdetr', 'yolov8', 'yolov5'),
    index=1
)

mode_selectbox = st.sidebar.selectbox(
    'Choose mode',
    ('Camera', 'Video'),
    index=0
)

video_source = mode_dict.get(mode_selectbox)
if mode_selectbox == 'Video':
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_source = tfile.name
    else:
        video_source = VIDEO_PATH

run = st.sidebar.checkbox("Start")

model = YOLO(model_dict.get(model_selectbox))

left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Original")

with right_column:
    st.subheader("Detection")

original_frame = left_column.image([])
pred_frame = right_column.image([])

cap = cv2.VideoCapture(video_source)

metrics_placeholder = st.empty()
try:
    while run:
        ret, frame = cap.read()

        if not ret:
            break
            
        original_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        start = time.time()
        
        # frame = cv2.resize(frame)

        results = model(frame)
        
        fps = 1 / (time.time() - start)
        latency_ms = results[0].speed['inference']
        
        device_type = "GPU" if model.device.type == "cuda" else "CPU"
        
        annotated = results[0].plot()

        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )
        
        metrics_placeholder.markdown(f"**FPS:** {fps:.2f} &nbsp;|&nbsp; **Latency:** {latency_ms:.2f} ms &nbsp;|&nbsp; **Device:** {device_type}")
        
        original_frame.image(
            original_rgb,
            channels="RGB",
            width='stretch'
        )

        pred_frame.image(
            annotated_rgb,
            channels="RGB",
            width='stretch'
        )
except Exception as e:
    print(f"Bị ngắt kết nối hoặc có lỗi: {e}")

finally:
    if cap is not None:
        cap.release()