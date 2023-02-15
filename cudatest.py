# import torch
# flag = torch.cuda.is_available()
# if flag:
#     print("CUDA可使用")
# else:
#     print("CUDA不可用")
#
# ngpu= 1
# # Decide which device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# print("驱动为：",device)
# print("GPU型号： ",torch.cuda.get_device_name(0))
import sys
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QSlider
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a media player object
        self.mediaPlayer = QMediaPlayer()

        # Set up the video widget
        videoWidget = QVideoWidget()
        self.mediaPlayer.setVideoOutput(videoWidget)

        # Create a layout to hold the video widget and progress slider
        layout = QHBoxLayout()
        layout.addWidget(videoWidget)
        self.slider = QSlider(Qt.Horizontal)
        layout.addWidget(self.slider)

        # Create a central widget to hold the layout
        centralWidget = QLabel()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        # Connect the slider to the media player
        self.slider.sliderMoved.connect(self.setPosition)
        self.mediaPlayer.positionChanged.connect(self.updateSlider)
        self.mediaPlayer.durationChanged.connect(self.updateDuration)

        # Load a media file
        self.mediaPlayer.setSource(r"E:\YOLO\yolov7_ship\testvideo.mp4")

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def updateSlider(self, position):
        self.slider.setValue(position)

    def updateDuration(self, duration):
        self.slider.setRange(0, duration)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
