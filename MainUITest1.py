import sys

from YoloV8 import *

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
           
def main():
    app = QApplication(sys.argv)

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

    # Create a CameraWidget instance and add it to the camera_tab_widget
    
    camera_widget = CameraWidget(0)
    camera_tab_widget.addTab(camera_widget, "Camera 1")

    '''camera2_panel = CameraWidget(1)  # Camera 2 with OpenCV initialization
    camera_tab_widget.addTab(camera2_panel, "Camera 2")

    camera3_panel = CameraWidget(0)  # Camera 3 with OpenCV initialization
    camera_tab_widget.addTab(camera3_panel, "Camera 3")

    camera4_panel = CameraWidget(3)  # Camera 3 with OpenCV initialization
    camera_tab_widget.addTab(camera4_panel, "Camera 4")'''

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
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
