import numpy as np
import cv2
import numpy as np
import argparse
import numpy as np
import supervision as sv
import torch

from ultralytics import YOLO

from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

ZONE_POLYGON = np.array([
    [0, 0],
    [640 // 2, 0],
    [640 // 2, 640 // 2],
    [0, 720 // 2]
])

zone_polygon = (ZONE_POLYGON * np.array([640, 640])).astype(int)

# Enable OpenCL for OpenCV (optional)
cv2.setUseOptimized(True)
cv2.ocl.setUseOpenCL(True)

parser = argparse.ArgumentParser(description="YOLOv8 live")
parser.add_argument("--webcam-resolution", default=[640, 640], nargs=2, type=int)
args = parser.parse_args()

zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.red,
    thickness=2,
    text_thickness=4,
    text_scale=2
)

class CameraWidget(QWidget):

    CameraResolutionWidth = 640
    CameraResolutionHeight = 640

    def __init__(self, camera_index, parent=None):
        super().__init__(parent)

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CameraResolutionWidth)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CameraResolutionHeight)

        # Load YOLO model
        self.model = YOLO("yolov8n.pt")
        self.model.predict(source=0, save=False, imgsz=640, conf=0.5, device='cuda:0',stream=True)
        self.model.device = torch.device("cuda:0")
        self.model = torch.cuda.set_per_process_memory_fraction(fraction=1.0, device='cuda:0')
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            print("CUDA is available.")
            self.device = torch.device("cuda:0")
            self.model.to(self.device)
        else:
            print("CUDA is not available. Using CPU.")
        #self.device = torch.device("cpu")

        self.print(torch.cuda.is_available())
        self.print(torch.cuda.get_device_name(0))
        self.print(torch.cuda.get_device_capability(0))
        self.print(torch.cuda.get_device_properties(0))
        self.print(torch.cuda.get_arch_list()[0])
        self.print(torch.cuda.get_allocator_backend())

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center align the image label
        self.image_label.setCursor(Qt.CursorShape.CrossCursor)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.image_label)
        self.layout.setContentsMargins(0, 0, 0, 0)  # Set margins to 0 to ensure the video feed occupies the entire space

        if not self.cap.isOpened():
            self.show_no_signal()
            return

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.frame_counter = 0
        self.start_time = cv2.getTickCount()

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

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Increment frame counter
            self.frame_counter += 1

            # Convert frame to PyTorch tensor and move to GPU
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frame_tensor = frame_tensor.to(self.device)
            print("Frame tensor device:", frame_tensor.device)

            # Perform inference
            result = self.model(frame_tensor, agnostic_nms=True)[0].cpu().numpy()
            detections = sv.Detections.from_yolov8(result)

            labels = [
                f"{self.model.names[class_id]} {confidence:0.2f}"
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

            # Draw FPS counter on the frame
            end_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_time - self.start_time)
            cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Update start time for next FPS calculation
            self.start_time = cv2.getTickCount()

            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to QImage
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format.Format_RGB888)

            # Display the QImage on the image_label
            self.image_label.setPixmap(QPixmap.fromImage(image))
            self.image_label.update()



if __name__ == "__main__":
    app = QApplication([])
    widget = CameraWidget(camera_index=0)
    widget.show()
    app.exit(app.exec_())
