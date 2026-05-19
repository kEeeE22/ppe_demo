YOLOV8_PATH = 'checkpoint\yolov8_womosaic.pt'
YOLOV5_PATH = 'checkpoint\yolov5l_epoch.pt'
RTDETR_PATH = r'checkpoint\rtdetr_womosaic.pt'

VIDEO_PATH = 'videos\YTDown_Shorts_construction-tap-trung-tbm-dau-gio-o-du-_Media_0dlE0Gv8ONE_001_1080p.mp4'

model_dict = {
    'rtdetr': RTDETR_PATH,
    'yolov8': YOLOV8_PATH,
    'yolov5': YOLOV5_PATH
}

mode_dict = {
    'Camera': 0,
    'Video': VIDEO_PATH
}