import streamlit as st
import cv2
from ultralytics import YOLO
import time
import tempfile
import os
# Giả sử bạn import config chứa model_dict và mode_dict
from config import * 

# 1. Caching Model để tránh reload khi Streamlit rerun
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

st.set_page_config(page_title="PPE Detection Demo", layout="wide")
st.title("Phát hiện Trang thiết bị Bảo hộ (PPE) Thời gian thực")

st.sidebar.header('Cấu hình Model')
model_selectbox = st.sidebar.selectbox(
    'Chọn mô hình',
    ('rtdetr', 'yolov8', 'yolov5'),
    index=1
)

# Load model với caching
model = load_model(model_dict.get(model_selectbox))

# Thêm Sliders cho Hyperparameters
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)
iou_threshold = st.sidebar.slider("NMS IoU Threshold", 0.0, 1.0, 0.45, 0.05)

st.sidebar.header('Cấu hình Nguồn phát')
mode_selectbox = st.sidebar.selectbox(
    'Chọn nguồn',
    ('Camera', 'Video'),
    index=0
)

video_source = mode_dict.get(mode_selectbox)
uploaded_file_path = None

if mode_selectbox == 'Video':
    uploaded_file = st.sidebar.file_uploader("Tải lên một file video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_source = tfile.name
        uploaded_file_path = tfile.name # Lưu lại để xóa sau
    else:
        video_source = VIDEO_PATH

run = st.sidebar.checkbox("BẮT ĐẦU PHÂN TÍCH")

# Bố cục UI
metrics_col, counts_col = st.columns([2, 1])
with metrics_col:
    metrics_placeholder = st.empty()
with counts_col:
    counts_placeholder = st.empty()

st.markdown("---")
left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Video Gốc")
    original_frame = st.empty()
with right_column:
    st.subheader("Kết quả Nhận diện")
    pred_frame = st.empty()

cap = cv2.VideoCapture(video_source)

try:
    while run:
        ret, frame = cap.read()
        if not ret:
            st.info("Đã phát hết video hoặc không tìm thấy nguồn.")
            break
            
        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start = time.time()
        
        # Inference với tham số từ UI
        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        fps = 1 / (time.time() - start)
        latency_ms = results[0].speed['inference']
        device_type = "GPU" if model.device.type == "cuda" else "CPU"
        
        # Đếm số lượng object (Giả sử labels: 0: Helmet, 1: Vest, 2: No_Helmet, v.v.)
        # Bạn có thể tuỳ chỉnh đoạn này theo file training data (data.yaml) của bạn
        class_names = results[0].names
        detected_classes = results[0].boxes.cls.cpu().numpy()
        counts = {}
        for cls_id in detected_classes:
            name = class_names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1
            
        # Vẽ boxes
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Cập nhật Metrics
        metrics_placeholder.markdown(
            f"**FPS:** `{fps:.1f}` &nbsp;|&nbsp; **Latency:** `{latency_ms:.1f} ms` &nbsp;|&nbsp; **Thiết bị:** `{device_type}`"
        )
        
        # Cập nhật bảng đếm
        count_str = " | ".join([f"**{k}:** {v}" for k, v in counts.items()])
        counts_placeholder.markdown(f"📊 **Thống kê:** {count_str}" if count_str else "📊 **Thống kê:** Chưa phát hiện")
        
        # Hiển thị hình ảnh
        original_frame.image(original_rgb, channels="RGB", use_container_width=True)
        pred_frame.image(annotated_rgb, channels="RGB", use_container_width=True)

except Exception as e:
    st.error(f"Đã xảy ra lỗi trong quá trình chạy: {e}")

finally:
    if cap is not None:
        cap.release()
    # Dọn dẹp file tạm
    if uploaded_file_path is not None and os.path.exists(uploaded_file_path):
        try:
            os.remove(uploaded_file_path)
        except Exception as e:
            pass