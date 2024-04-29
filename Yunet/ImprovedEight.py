import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
from yunet import YuNet

# Initialize a dictionary to store the tracking history of objects
track_history = defaultdict(list)

# Set OpenCV backend to use hardware acceleration if available
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Adjust the number of threads for optimal performance

# Load the YOLO model
try:
    yolo_model = YOLO("yolov8n.yaml")
    yolo_model = YOLO("yolov8n.pt")
    yolo_names = yolo_model.model.names
except Exception as e:
    print("Error loading YOLO model:", e)
    exit(1)

# Load the YuNet model
try:
    yunet_model = YuNet(modelPath='YuNet_model.onnx',
                        inputSize=[640, 480],
                        confThreshold=0.9,
                        nmsThreshold=0.3,
                        topK=5000,
                        backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                        targetId=cv2.dnn.DNN_TARGET_CPU)
except Exception as e:
    print("Error loading YuNet model:", e)
    exit(1)

# Check if GPU is available
device = torch.device("cuda")

# Flag to indicate if an object is selected for tracking
object_selected = False
selected_track_id = None

# Mouse callback function
def select_object(event, x, y, flags, param):
    global object_selected, selected_track_id
    if event == cv2.EVENT_LBUTTONDOWN:
        object_selected = True
        selected_track_id = None

# Open the default camera
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error accessing camera"

# Get the default camera resolution
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Adjust FPS based on camera's actual FPS

# Create a window and set mouse callback function
cv2.namedWindow("Object Tracking")
cv2.setMouseCallback("Object Tracking", select_object)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Perform object detection and tracking with YOLOv8
        with torch.no_grad():
            yolo_detections = yolo_model.track(frame, device=device, persist=True, verbose=False)
        yolo_boxes = yolo_detections[0].boxes.xyxy.cpu()

        if yolo_detections[0].boxes.id is not None:
            # Extract YOLO prediction results
            yolo_clss = yolo_detections[0].boxes.cls.cpu().tolist()
            yolo_track_ids = yolo_detections[0].boxes.id.int().cpu().tolist()
            yolo_confs = yolo_detections[0].boxes.conf.float().cpu().tolist()

            # Create an Annotator object for drawing bounding boxes and labels
            annotator = Annotator(frame, line_width=3)

            for yolo_box, yolo_cls, yolo_track_id in zip(yolo_boxes, yolo_clss, yolo_track_ids):
                # Determine the color of the bounding box
                color = colors(int(yolo_cls), True)

                # If the object is selected, change the color to yellow
                if object_selected and selected_track_id == yolo_track_id:
                    color = (0, 255, 255)  # Yellow color

                # Draw bounding boxes and labels
                annotator.box_label(yolo_box, color=color, label=yolo_names[int(yolo_cls)])

                # Store tracking history
                yolo_track = track_history[yolo_track_id]
                yolo_track.append(((yolo_box[0] + yolo_box[2]) // 2, (yolo_box[1] + yolo_box[3]) // 2))

                # Limit the history to 30 points
                if len(yolo_track) > 30:
                    yolo_track.pop(0)

                # Draw the track history
                yolo_points = (int(yolo_track[-1][0]), int(yolo_track[-1][1]))  # Get the latest point
                cv2.circle(frame, yolo_points, 7, color, -1)

        # Perform facial detection with YuNet
        yunet_faces = yunet_model.infer(frame)

        # Draw bounding boxes for detected faces
        for bbox in yunet_faces:
            bbox = bbox.astype(np.int32)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)


        # Draw bounding boxes for detected faces
        for yunet_face in yunet_faces:
            bbox = yunet_face[0:4].astype(np.int32)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Display the frame with annotations
        cv2.imshow("Object Tracking", frame)

        # Delay to match the frame rate
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    else:
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
