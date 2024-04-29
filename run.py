import cv2
import argparse
import numpy as np
import time

from yolov8 import YOLOv8
from yunet import YuNet

cam_width = 1280
cam_height = 720

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX, cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN, cv2.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--model', '-m', type=str, default='face_detection_yunet_2023mar_modified.onnx', help="Usage: Set model type, defaults to 'face_detection_yunet_2023mar_modified.onnx'.")
parser.add_argument('--backend_target', '-bt', type=int, default=0)
parser.add_argument('--conf_threshold', type=float, default=0.5, help='Usage: Set the minimum needed confidence for the model to identify a face, defaults to 0.9. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Usage: Suppress bounding boxes of IoU >= nms_threshold. Default = 0.3.')
parser.add_argument('--top_k', type=int, default=5000, help='Usage: Keep top_k bounding boxes before NMS.')
args = parser.parse_args()

def visualize(image, results, input_size, selected_index=None, box_color=(0, 208, 255), text_color=(0, 0, 255), fps=None, tracking_mode_text="", algorithm_name="", locked_index=None):
    output = image.copy()

    # Calculate FPS
    if fps:
        cv2.putText(output, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for idx, det in enumerate(results):
        bbox = det[0:4].astype(np.int32)
        bbox_original_size = (int(bbox[0] * image.shape[1] / input_size[0]),
                              int(bbox[1] * image.shape[0] / input_size[1]),
                              int(bbox[2] * image.shape[1] / input_size[0]),
                              int(bbox[3] * image.shape[0] / input_size[1]))

        if selected_index is not None and idx != selected_index:
            continue  # Skip drawing other bounding boxes if single-tracking is initiated

        if idx == locked_index:
            box_color = (0, 255, 0)  # Change bounding box color to green for the locked object

        cv2.rectangle(output, (bbox_original_size[0], bbox_original_size[1]),
                     (bbox_original_size[0] + bbox_original_size[2], bbox_original_size[1] + bbox_original_size[3]),
                     box_color, 2)

        conf = det[-1]
        cv2.putText(output, '{:.4f}'.format(conf), (bbox_original_size[0], bbox_original_size[1] - 5),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        
        # Display algorithm name beside the confidence level
        label = f"{algorithm_name}: {conf:.2f}"
        cv2.putText(output, label, (bbox_original_size[0] + bbox_original_size[2] + 5, bbox_original_size[1] + 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    # Draw text indicator for tracking mode
    cv2.putText(output, tracking_mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return output

backend_id = backend_target_pairs[args.backend_target][0]
target_id = backend_target_pairs[args.backend_target][1]

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

# Instantiate YuNet
model = YuNet(modelPath=args.model,
            inputSize=[640, 640],
            confThreshold=args.conf_threshold,
            nmsThreshold=args.nms_threshold,
            topK=args.top_k,
            backendId=backend_id,
            targetId=target_id)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
model.setInputSize([w, h])

# Initialize YOLOv8 object detector
model_path = "yolov8n_fmodel.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

selected_object_index = -1  # Initialize the index of the selected object
selected_object_bbox = None  # Initialize the selected object's bounding box
multi_tracking = True  # Flag to indicate multi-tracking is active by default
selected_algorithm = "YuNet"  # Initialize selected algorithm as YuNet
locked_object_index = None  # Initialize the index of the locked object during single-tracking

# Variables for FPS calculation
prev_time = 0
fps = 0

def toggle_tracking_mode():
    global multi_tracking
    multi_tracking = not multi_tracking

def select_object_with_arrow_keys(key):
    global selected_object_index
    if key == 0:  # UP arrow key
        selected_object_index -= 1
    elif key == 1:  # DOWN arrow key
        selected_object_index += 1

    # Ensure the selected object index stays within bounds
    selected_object_index = max(0, min(selected_object_index, len(yunet_results) - 1))

# Main loop
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Calculate FPS
    current_time = time.time()
    if prev_time != 0:
        fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Inference with selected algorithm (YuNet or YOLOv8)
    if selected_algorithm == "YuNet":
        # Inference with YuNet
        yunet_results = model.infer(frame)
        detection_results = yunet_results
    elif selected_algorithm == "YOLOv8":
        # Inference with YOLOv8
        yolo_boxes, _, _ = yolov8_detector(frame)
        detection_results = yolo_boxes

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('w'), ord('W'), 82]:  # UP arrow key
        select_object_with_arrow_keys(0)
    elif key in [ord('s'), ord('S'), 84]:  # DOWN arrow key
        select_object_with_arrow_keys(1)
    elif key == ord('e'):  # 'e' key
        if selected_object_index != -1:
            selected_object_bbox = yunet_results[selected_object_index][0:4].astype(np.int32)
            locked_object_index = selected_object_index  # Lock the selected object
    elif key == ord('y'):  # 'y' key for YuNet
        selected_algorithm = "YuNet"
    elif key == ord('o'):  # 'o' key for YOLOv8
        selected_algorithm = "YOLOv8"
    elif key == ord('t'):  # 't' key to toggle tracking mode
        toggle_tracking_mode()

    # Visualize detection results with selected object highlighted
    if multi_tracking or selected_object_index == -1:
        frame_with_bboxes = visualize(frame, detection_results, [w, h], tracking_mode_text="Multi-Tracking Mode", algorithm_name=selected_algorithm, fps=fps)
    else:
        frame_with_bboxes = visualize(frame, detection_results, [w, h], selected_index=selected_object_index, tracking_mode_text="Single-Tracking Mode", algorithm_name=selected_algorithm, locked_index=locked_object_index, fps=fps)

    # Display the combined image with detection results
    cv2.imshow("Detected Objects", frame_with_bboxes)

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
