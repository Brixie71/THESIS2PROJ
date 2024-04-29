import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt5.QtGui import QFont, QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt5 Main Window")
        self.setFixedSize(1920, 1080)  # Adjusted window size

        # Add a QLabel for the background image
        self.background_label = QLabel(self)
        self.background_label.setGeometry(0, 0, 1920, 1080)  # Adjusted size to match window size
        pixmap = QPixmap("bg.png")  # Replace "bg.png" with your image file path
        pixmap = pixmap.scaled(self.size())  # Scale pixmap to match window size
        self.background_label.setPixmap(pixmap)

        # Add a QPushButton
        self.start_button = QPushButton("Start", self)
        self.start_button.setGeometry(500, 600, 300, 150)  # Adjusted position and size
        self.start_button.setFont(QFont("Impact", 40))  # Increased font size for button text
        self.start_button.setStyleSheet("background-color: green; color: white; border-radius: 10px;")  # Set button color to green and make edges round
        self.start_button.clicked.connect(self.start_button_clicked)
        
        # Raise the button to the top layer
        self.start_button.raise_()

    def start_button_clicked(self):
        print("Start button clicked!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
