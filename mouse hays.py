import cv2
import argparse
import numpy as np
import time

from Algorithms import YOLOv8
from Algorithms.yunet import YuNet

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
parser.add_argument('--model', '-m', type=str, default='YuNet_model.onnx', help="Usage: Set model type, defaults to 'face_detection_yunet_2023mar_modified.onnx'.")
parser.add_argument('--backend_target', '-bt', type=int, default=1)
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

def mouse_callback(event, x, y, flags, param):
    global selected_object_index, detection_results  # Add detection_results to global variables

    if event == cv2.EVENT_LBUTTONDOWN:
        # Loop through detection results to find the closest object to the mouse click
        min_distance = float('inf')
        selected_object_index = -1
        for idx, det in enumerate(detection_results):
            bbox = det[0:4].astype(np.int32)
            bbox_center_x = bbox[0] + bbox[2] / 2
            bbox_center_y = bbox[1] + bbox[3] / 2
            distance = np.sqrt((x - bbox_center_x)**2 + (y - bbox_center_y)**2)
            if distance < min_distance:
                min_distance = distance
                selected_object_index = idx

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

# Initialize YuNet and YOLOv8 models
backend_id = backend_target_pairs[args.backend_target][0]
target_id = backend_target_pairs[args.backend_target][1]
yuNet_model = YuNet(modelPath=args.model,
              inputSize=[640, 640],
              confThreshold=args.conf_threshold,
              nmsThreshold=args.nms_threshold,
              topK=args.top_k,
              backendId=backend_id,
              targetId=target_id)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
yuNet_model.setInputSize([w, h])

model_path = "path/to/yolov8/model.pt"  # Provide the correct path to your YOLOv8 model
yolov8_model = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

selected_object_index = -1  # Initialize the index of the selected object
multi_tracking = True  # Flag to indicate multi-tracking is active by default
indicator_text = "No Selected Object"  # Text for indicating whether an object is selected or not

# Variables for FPS calculation
prev_time = 0
fps = 0

# Create the window
cv2.namedWindow("Detected Objects")

# Set the mouse callback function
cv2.setMouseCallback("Detected Objects", mouse_callback)

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

    # Inference with YuNet
    yunet_results = yuNet_model.infer(frame)

    # Inference with YOLOv8
    yolo_boxes, _, _ = yolov8_model(frame)

    # Combine detection results from both models
    combined_results = yunet_results + yolo_boxes

    # Visualize combined detection results
    frame_with_bboxes = visualize(frame, combined_results, [w, h], tracking_mode_text="Multi-Tracking Mode", algorithm_name="Combined", fps=fps)

    # Display the combined image with detection results
    cv2.putText(frame_with_bboxes, indicator_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Detected Objects", frame_with_bboxes)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
