import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget, QPushButton, QHBoxLayout, QGridLayout, QGroupBox, QTextEdit, QLabel
from PyQt5.QtGui import QFont, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

WindowWidth = 1366
WindowHeight = 768

class HighlightButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.default_color = QColor("#4CAF50")
        self.hover_color = QColor("#45a049")
        self.pressed_color = QColor("#357a38")
        self.setMouseTracking(True)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 24px;
                text-align: center;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 8px;
            }
        """)

    def enterEvent(self, event):
        if self.isEnabled():
            self.setStyleSheet("""
                QPushButton {
                    background-color: #45a049;
                    color: white;
                    border: none;
                    padding: 10px 24px;
                    text-align: center;
                    font-size: 16px;
                    margin: 4px 2px;
                    border-radius: 8px;
                }
            """)
            super().enterEvent(event)

    def leaveEvent(self, event):
        if self.isEnabled():
            self.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 24px;
                    text-align: center;
                    font-size: 16px;
                    margin: 4px 2px;
                    border-radius: 8px;
                }
            """)
            super().leaveEvent(event)

    def mousePressEvent(self, event):
        if self.isEnabled():
            self.setStyleSheet("""
                QPushButton {
                    background-color: #357a38;
                    color: white;
                    border: none;
                    padding: 10px 24px;
                    text-align: center;
                    font-size: 16px;
                    margin: 4px 2px;
                    border-radius: 8px;
                }
            """)
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.isEnabled():
            self.setStyleSheet("""
                QPushButton {
                    background-color: #45a049;
                    color: white;
                    border: none;
                    padding: 10px 24px;
                    text-align: center;
                    font-size: 16px;
                    margin: 4px 2px;
                    border-radius: 8px;
                }
            """)
            super().mouseReleaseEvent(event)

class CameraWidget(QWidget):
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.video_capture = cv2.VideoCapture(self.camera_id)
        
        # Create a QLabel to display the camera feed
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        
        # Create a layout and add the camera label to it
        layout = QVBoxLayout()
        layout.addWidget(self.camera_label)
        self.setLayout(layout)
        
        # Start a timer to update the camera feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)  # Update every 50 milliseconds
    
    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to QImage
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(image)
            # Update the QLabel with the new frame
            self.camera_label.setPixmap(pixmap)

def center_window(window):
    screen_resolution = QApplication.desktop().screenGeometry()
    screen_width, screen_height = screen_resolution.width(), screen_resolution.height()
    x = (screen_width - WindowWidth) // 2
    y = (screen_height - WindowHeight) // 2
    window.move(x, y)
           
def main():
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

    CAM1 = 0  # Assuming the camera ID is 0 for testing purposes
    
    # Manually creating camera tabs
    camera1_panel = CameraWidget(CAM1)  # Camera 1 with OpenCV initialization
    camera_tab_widget.addTab(camera1_panel, "Camera 1")
    
    # Create a horizontal group container
    horizontal_group_container = QGroupBox()
    horizontal_layout = QHBoxLayout(horizontal_group_container)
    horizontal_group_container.setFixedHeight(200)

    # Create two group containers inside the horizontal container
    grid_group_box = QGroupBox("Camera Controls")

    grid_layout = QGridLayout(grid_group_box)

    grid_group_box.setFixedHeight(200)
    grid_group_box.setAutoFillBackground(True)
    grid_group_box.setFixedWidth(500)

    # Define functions for button actions
    def single_tracking(self):
        self.single_tracking_mode = not self.single_tracking_mode
        if self.single_tracking_mode:   
            print("Single Tracking Mode Enabled")
    
    def multi_tracking():
        print("Multi Tracking")

    def select_left():
        print("Select Left")

    def select_right(self):
        self.selected_object_index += 1

    def target_lock(self):
       self.selected_object_bbox = self.results[self.selected_object_index][0:4].astype(np.int32)

    def disengage():
        print("Dis-Engage")

    def face_detection():
        pass  # Add your face detection code here

    def object_detection():
        pass  # Add your object detection code here

    # Create individual buttons and add them to the grid layout
    buttons = [
        HighlightButton("Single Tracking"),
        HighlightButton("Multi Tracking"),
        HighlightButton("<- Select"),
        HighlightButton("Select ->"),
        HighlightButton("Target Lock"),
        HighlightButton("Dis-Engage"),
        HighlightButton("Face Detection"),
        HighlightButton("Object Detection")
    ]

    # Connect each button's clicked signal to its respective function
    buttons[0].clicked.connect(single_tracking)
    buttons[1].clicked.connect(multi_tracking)
    buttons[2].clicked.connect(select_left)
    buttons[3].clicked.connect(select_right)
    buttons[4].clicked.connect(target_lock)
    buttons[5].clicked.connect(disengage)
    buttons[6].clicked.connect(face_detection)
    buttons[7].clicked.connect(object_detection)

    for i, button in enumerate(buttons):
        row = i // 2
        col = i % 2
        grid_layout.addWidget(button, row, col)

    layout2 = QVBoxLayout(grid_group_box)

    grid_group_box.setLayout(grid_layout)
    layout.addWidget(grid_group_box, alignment=Qt.AlignCenter)

    window.show()
    main_application.exec_()

if __name__ == '__main__':
    main()
