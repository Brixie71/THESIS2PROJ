import sys
import cv2
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget, QTextEdit, QHBoxLayout, QGroupBox
from PyQt5.QtGui import QFont

WindowWidth = 1366
WindowHeight = 768

def center_window(window):
    screen_resolution = QApplication.desktop().screenGeometry()
    screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
    x = (screen_width - WindowWidth) // 2
    y = (screen_height - WindowHeight) // 2
    window.move(x, y)

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
           "scissors", "teddy bear", "hair drier", "toothbrush"]

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, scale):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
        scale (float): Scale factor used during preprocessing.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (int(x * scale), int(y * scale)), (int(x_plus_w * scale), int(y_plus_h * scale)), color, 2)
    cv2.putText(img, label, (int(x * scale) - 10, int(y * scale) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
           
def main(onnx_model):
    main_application = QApplication(sys.argv)

    font = QFont("Cambria")
    window = QMainWindow()
    window.setWindowTitle("Brion Tactical Systems - GROUP 5 FINAL THESIS")
    window.setGeometry(0, 0, WindowWidth, WindowHeight)
    window.setMinimumWidth(WindowWidth)
    window.setMinimumHeight(WindowHeight)
    window.setFont(font)
    window.setAutoFillBackground(True)
   
    center_window(window)

    central_widget = QWidget(window)
    central_widget.setStyleSheet('''background-color: #f8f9fa;
                                    color: Black;
                                 ''')
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)

    # Load the ONNX model
    model = cv2.dnn.readNetFromONNX(onnx_model)

    # Open camera capture
    cap = cv2.VideoCapture(0)

    # Initialize FPS counter
    fps = cv2.TickMeter()

    while cap.isOpened():
        # Start FPS counter
        fps.start()

        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            break

        [height, width, _] = frame.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = frame

        # Calculate scale factor
        scale = length / 640

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        model.setInput(blob)

        # Perform inference
        outputs = model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            draw_bounding_box(
                frame,
                class_ids[index],
                scores[index],
                box[0],
                box[1],
                box[0] + box[2],
                box[1] + box[3],
                scale
            )

        # Stop FPS counter
        fps.stop()

        # Calculate and display FPS
        cv2.putText(frame, f"FPS: {fps.getFPS():.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        


    camera_tab_widget = QTabWidget()
    
    camera_tab_widget.setStyleSheet("""
                                    background-color: dark-gray;
                                    margin: -1;
                                    padding: -1;
                                    font-size: 20px;
                                    color: #000000;
                                    text-align: Center;
                                    """)
    layout.addWidget(camera_tab_widget)

    CCTV_IP = 'rtsp://admin:1Rolando23@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif'

    # Manually creating camera tabs
    camera1_panel = CameraWidget(0)  # Camera 1 with OpenCV initialization
    camera_tab_widget.addTab(camera1_panel, "Camera 1")
    
    camera2_panel = CameraWidget(1)  # Camera 2 with OpenCV initialization
    camera_tab_widget.addTab(camera2_panel, "Camera 2")

    camera3_panel = CameraWidget(3)  # Camera 3 with OpenCV initialization
    camera_tab_widget.addTab(camera3_panel, "Camera 3")

    camera4_panel = CameraWidget(3)  # Camera 3 with OpenCV initialization
    camera_tab_widget.addTab(camera4_panel, "Camera 4")

    # Create a horizontal group container
    horizontal_group_container = QGroupBox()
    horizontal_layout = QHBoxLayout(horizontal_group_container)
    horizontal_group_container.setFixedHeight(200)

    # Create two group containers inside the horizontal container
    group_container1 = QGroupBox("Camera Logs")
    group_container2 = QGroupBox("Camera Status")

    # Set fixed height for the group containers
    group_container1.setFixedHeight(200)
    group_container1.setAutoFillBackground(True)

    group_container2.setFixedHeight(200)
    group_container2.setAutoFillBackground(True)
    group_container2.setFixedWidth(400)

    # Add a text box to each group container
    text_edit1 = QTextEdit()
    text_edit2 = QTextEdit()

    layout1 = QVBoxLayout(group_container1)
    layout2 = QVBoxLayout(group_container2)

    layout1.addWidget(text_edit1)
    layout2.addWidget(text_edit2)

    # Add the group containers to the horizontal layout
    horizontal_layout.addWidget(group_container1)
    horizontal_layout.addWidget(group_container2)

    # Add the horizontal group container to the main layout
    layout.addWidget(horizontal_group_container)

    window.show()
    main_application.exec_()

if __name__ == '__main__':
    main()
