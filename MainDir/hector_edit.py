import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget, QTextEdit, QHBoxLayout, QGroupBox
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt

WindowWidth = 1366
WindowHeight = 768

class CameraWidget(QWidget):
    from PyQt5.QtWidgets import QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer

def __init__(self, camera_id, parent=None):
        super(CameraWidget, self).__init__(parent)
        self.camera_id = camera_id
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        self.selected_object_index = -1
        self.multi_tracking = True
        self.locked_object_index = None

        # Set up OpenCV capture
        self.capture = cv2.VideoCapture(camera_id)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 / 30)  # Update at 30 FPS

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            x = event.x()
            y = event.y()
            # Implement object selection logic here
            pass

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

    camera1_panel = CameraWidget(0)
    camera_tab_widget.addTab(camera1_panel, "Camera 1")

    camera2_panel = CameraWidget(1)
    camera_tab_widget.addTab(camera2_panel, "Camera 2")

    camera3_panel = CameraWidget(2)
    camera_tab_widget.addTab(camera3_panel, "Camera 3")

    horizontal_group_container = QGroupBox()
    horizontal_layout = QHBoxLayout(horizontal_group_container)
    horizontal_group_container.setFixedHeight(200)

    group_container1 = QGroupBox("Camera Logs")
    group_container2 = QGroupBox("Camera Status")

    group_container1.setFixedHeight(200)
    group_container1.setAutoFillBackground(True)

    group_container2.setFixedHeight(200)
    group_container2.setAutoFillBackground(True)
    group_container2.setFixedWidth(400)

    text_edit1 = QTextEdit()
    text_edit2 = QTextEdit()

    layout1 = QVBoxLayout(group_container1)
    layout2 = QVBoxLayout(group_container2)

    layout1.addWidget(text_edit1)
    layout2.addWidget(text_edit2)

    horizontal_layout.addWidget(group_container1)
    horizontal_layout.addWidget(group_container2)

    layout.addWidget(horizontal_group_container)

    window.show()
    main_application.exec_()

if __name__ == '__main__':
    main()
