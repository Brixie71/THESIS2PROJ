# Adaptive Object Tracking and Following using YoloV8 and Yunet Fast-RCNN Algorithms
This Thesis project was done via a Group Thesis. 

The objective of this project is to track people's Physical Appearance across 3 CCTV Cameras. The System is requires human intervention for Identification and continuous tracking of the Subject. The System uses YoloV8 to Track the Physical Appearance of the Person and uses the YuNet Algorithm to Track the faces of the persons selected by the User in the software. 

# Role 
The Group was composed of 4 People. I was the Assistant Developer and Tester of the Project. I was included in Debugging and Trial and Error of the System. It was one of the Hardest Systems I've Worked on 

# OpenCV
The System uses Open CV Library via the Python Programming Language. The YoloV8 and YuNet Algorithm was Implemented using the Python Programming language.

# Roboflow
The Dataset was trained in Roboflow, aquiring a trained model to track persons and specific objects that matches the description of the selected image from the 3 CCTV Cameras.

# Downside 
- Even the Slightest Difference in Lighting, Brightness, Exposure, Contrast, Hue, and ISO, will disengage tracking of persons and objects selected when comparing persons and objects visible from the 3 CCTV Cameras.
- Combining YoloV8 and YuNet Fast RCNN Algorithm made Object Detection Slower, thus requiring to use an NVidia Graphics card with the Use of the NVidia's CUDA Cores to Optimize the Performance of the Detection. Perforamnce of the System is slow, ranging from 15 to 25 FPS due to the simultanious object detection of YoloV8 and YuNet FRCCN.
 
