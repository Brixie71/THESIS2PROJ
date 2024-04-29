import argparse
import cv2
import numpy as np
import YoloV8

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
    """
    Main function to load ONNX model, perform inference on camera feed, draw bounding boxes, display the output image,
    and calculate FPS.

    Args:
        onnx_model (str): Path to the ONNX model.

    Returns:
        list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
    """

    # Load the ONNX model
    #model = cv2.dnn.readNetFromONNX(onnx_model)

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
        YoloV8.model.setInput(blob)

        # Perform inference
        outputs = YoloV8.model.forward()

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

        # Display the image with bounding boxes and FPS
        cv2.imshow("Camera Feed", frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.onnx", help="Input your ONNX model.")
    args = parser.parse_args()
    main(args.model)
