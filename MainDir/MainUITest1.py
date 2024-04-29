import sys
from YoloV8 import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget, QPushButton, QHBoxLayout, QGridLayout, QGroupBox, QTextEdit
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt

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

    CAM1 = 'rtsp://admin:1Rolando23@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=RTSP'
    CAM2 = 'rtsp://admin:1Rolando23@192.168.1.109:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=RTSP'
    CAM3 = 'rtsp://admin:1Rolando23@192.168.1.110:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=RTSP'
    CAMTEST = 'http://admin:1Rolando23@192.168.1.110:80/cam/realmonitor?channel=1&subtype=0&unicast=true'

    # Manually creating camera tabs
    camera1_panel = CameraWidget(1)  # Camera 1 with OpenCV initialization
    camera_tab_widget.addTab(camera1_panel, "Camera 1")
    
    camera2_panel = CameraWidget(2)  # Camera 2 with OpenCV initialization
    camera_tab_widget.addTab(camera2_panel, "Camera 2")

    camera3_panel = CameraWidget(0)  # Camera 3 with OpenCV initialization
    camera_tab_widget.addTab(camera3_panel, "Camera 3")

    # Create a horizontal group container
    horizontal_group_container = QGroupBox()
    horizontal_layout = QHBoxLayout(horizontal_group_container)
    horizontal_group_container.setFixedHeight(200)

    # Create two group containers inside the horizontal container
    group_container1 = QGroupBox("Camera Logs")
    # Button 
    grid_group_box = QGroupBox("Camera Controls")

    # Remove existing layout from group_container1
    group_container1.setLayout(QVBoxLayout())

    grid_layout = QGridLayout(grid_group_box)

    # Set fixed height for the group containers
    group_container1.setFixedHeight(200)
    group_container1.setAutoFillBackground(True)

    grid_group_box.setFixedHeight(200)
    grid_group_box.setAutoFillBackground(True)
    grid_group_box.setFixedWidth(500)

    # Add a text box to each group container
    text_edit1 = QTextEdit()

    results = [] # Initialize an empty list for results
    selected_object_index = 0


    # Ensure the selected object index stays within bounds
    selected_object_index = max(0, min(selected_object_index, len(results) - 1))

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

    # Create individual buttons and add them to the grid layout
    buttons = [
        HighlightButton("Single Tracking"),
        HighlightButton("Multi Tracking"),
        HighlightButton("<- Select"),
        HighlightButton("Select ->"),
        HighlightButton("Target Lock"),
        HighlightButton("Dis-Engage")
    ]

  # Connect each button's clicked signal to its respective function
    buttons[0].clicked.connect(single_tracking)
    buttons[1].clicked.connect(multi_tracking)
    buttons[2].clicked.connect(select_left)
    buttons[3].clicked.connect(select_right)
    buttons[4].clicked.connect(target_lock)
    buttons[5].clicked.connect(disengage)

    for i, button in enumerate(buttons):
        row = i // 2
        col = i % 2
        grid_layout.addWidget(button, row, col)

    layout1 = QVBoxLayout(group_container1)
    layout2 = QVBoxLayout(grid_group_box)

    grid_group_box.setLayout(grid_layout)
    layout1.addWidget(text_edit1)
    layout2.addWidget(grid_group_box)

    # Add the group containers to the horizontal layout
    horizontal_layout.addWidget(group_container1)
    horizontal_layout.addWidget(grid_group_box)

    # Add the horizontal group container to the main layout
    layout.addWidget(horizontal_group_container)

    window.show()
    main_application.exec_()

if __name__ == '__main__':
    main()
