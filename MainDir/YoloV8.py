import numpy as np
import cv2
import argparse
import numpy as np
import supervision as sv
import torch

from ultralytics import YOLO
from yunet import YuNet
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX, cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN, cv2.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--model', '-m', type=str, default='YuNet_model.onnx',
                    help="Usage: Set model type, defaults to 'face_detection_yunet_2023mar_modified.onnx'.")
parser.add_argument('--backend_target', '-bt', type=int, default=1)
parser.add_argument('--conf_threshold', type=float, default=0.9,
                    help='Usage: Set the minimum needed confidence for the model to identify a face, defaults to 0.9. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.1,
                    help='Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3.')
parser.add_argument('--top_k', type=int, default=5000,
                    help='Usage: Keep top_k bounding boxes before NMS.')
args = parser.parse_args()

def visualize(image, results, input_size, box_color=(0, 208, 255), text_color=(0, 0, 255), fps=None):
    output = image.copy()

    for det in results:
        bbox = det[0:4].astype(np.int32)
        bbox_original_size = (int(bbox[0] * image.shape[1] / input_size[0]),
                              int(bbox[1] * image.shape[0] / input_size[1]),
                              int(bbox[2] * image.shape[1] / input_size[0]),
                              int(bbox[3] * image.shape[0] / input_size[1]))

        cv2.rectangle(output, (bbox_original_size[0], bbox_original_size[1]),
                     (bbox_original_size[0] + bbox_original_size[2], bbox_original_size[1] + bbox_original_size[3]),
                     box_color, 2)

        conf = det[-1]
        cv2.putText(output, '{:.4f}'.format(conf), (bbox_original_size[0], bbox_original_size[1] - 5),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    return output

# Instantiate YuNet
YuNet.model = YuNet(modelPath=args.model,
                    inputSize=[640, 480],
                    confThreshold=args.conf_threshold,
                    nmsThreshold=args.nms_threshold,
                    topK=args.top_k,
                    backendId=backend_target_pairs[args.backend_target][0],
                    targetId=backend_target_pairs[args.backend_target][1])

# Load YOLO model
model = YOLO("yolov8n.pt")

class CameraWidget(QWidget):
    def __init__(self, camera_index, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.image_label)
        self.layout.setContentsMargins(0, 0, 0, 0)
        if not self.cap.isOpened():
            self.show_no_signal()
            return
        self.frame_counter = 0
        self.start_time = cv2.getTickCount()

    def process_frames(self):
        tm = cv2.TickMeter()
        ret, frame = self.cap.read()
        if ret:
            tm.start()
            YuNetResults = YuNet.model.infer(frame)  # YuNet inference
            results = model(frame, agnostic_nms=True)[0]  # YOLO inference

            detections = sv.Detections.from_yolov8(results)
            tm.stop()
            labels = [
                f"{model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]

            box_annotator = sv.BoxAnnotator(
                    thickness=2,
                    text_thickness=2,
                    text_scale=0.8)
            
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections, 
                labels=labels
            )

            # Draw results on the input image
            frame = visualize(frame, YuNetResults, input_size=[640, 480])

            # Draw FPS counter on the frame
            end_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_time - self.start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
            self.start_time = cv2.getTickCount()

            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to QImage
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format.Format_RGB888)

    def show_no_signal(self):
        self.image_label.setText("<b>NO SIGNAL</b>")
        self.image_label.setStyleSheet("""
                                        color: #f8f9fa;
                                        font-family: Calibri, serif;
                                        font-size: 50px;
                                        font-weight: bold;
                                        border: 2px solid black;
                                        border-radius: 0px;
                                        padding: 10px;
                                        margin: 5px;
                                        opacity: 0.5;
                                        text-align: center;
                                        text-decoration: underline;
                                        """)



