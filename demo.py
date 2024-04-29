import argparse

import numpy as np
import cv2

cv2.setUseOptimized(True)
cv2.ocl.setUseOpenCL(True)

class YuNet:
    def __init__(self, modelPath, inputSize=[640, 640], confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv2.FaceDetectorYN.create(
                    model=self._modelPath,
                    config="",
                    input_size=self._inputSize,
                    score_threshold=self._confThreshold,
                    nms_threshold=self._nmsThreshold,
                    top_k=self._topK,
                    backend_id=self._backendId,
                    target_id=self._targetId)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        # Forward
        faces = self._model.detect(image)
        return np.array([]) if faces[1] is None else faces[1]

# Valid combinations of backends and targets
backend_target_pairs = [[cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],[cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],[cv2.dnn.DNN_BACKEND_CUDA,   
                         cv2.dnn.DNN_TARGET_CUDA_FP16], [cv2.dnn.DNN_BACKEND_TIMVX, cv2.dnn.DNN_TARGET_NPU],[cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]]

parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--model', '-m', type=str, default='YuNet_model.onnx',
                    help="Usage: Set model type, defaults to 'face_detection_yunet_2023mar_modified.onnx'.")
parser.add_argument('--backend_target', '-bt', type=int, default=0)
parser.add_argument('--conf_threshold', type=float, default=0.9,
                    help='Usage: Set the minimum needed confidence for the model to identify a face, defaults to 0.9. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3,
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

backend_id = backend_target_pairs[args.backend_target][0]
target_id = backend_target_pairs[args.backend_target][1]

# Instantiate YuNet
model = YuNet(modelPath=args.model,
            inputSize=[320, 320],
            confThreshold=args.conf_threshold,
            nmsThreshold=args.nms_threshold,
            topK=args.top_k,
            backendId=backend_id,
            targetId=target_id)

if __name__ == '__main__':
    
    # Omit input to call default camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model.setInputSize([w, h])

    tm = cv2.TickMeter()
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        results = model.infer(frame) # results is a tuple
        tm.stop()

        # Draw results on the input image
        frame = visualize(frame, results, input_size=[w, h], fps=tm.getFPS())

        # Visualize results in a new Window
        cv2.imshow('YuNet Demo', frame)

        tm.reset()
