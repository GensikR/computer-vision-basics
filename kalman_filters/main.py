import sys
import argparse
import numpy as np
from PySide2 import QtCore, QtWidgets, QtGui
import cv2
from MotionDetector import MotionDetector
from KalmanFilter import KalmanFilter

class MainWindow(QtWidgets.QWidget):
    def __init__(self, video_path, motion_detector):
        """
        Initialize the main application window.

        Parameters:
        - video_path: Path to the video file.
        - motion_detector: Instance of the MotionDetector class to track motion in video frames.
        """
        super().__init__()

        self.video_path = video_path
        self.motion_detector = motion_detector
        self.frames = []  # List to hold video frames
        self.current_frame = 0  # Index of the current frame being displayed
        self.playing = False  # Flag to indicate if video is playing
        self.timer = QtCore.QTimer()  # Timer to control video playback

        # Load video frames from the file
        self.load_video()

        # Initialize UI components
        self.init_ui()

        # Connect the timer's timeout signal to the frame update function
        self.timer.timeout.connect(self.update_frame)

    def load_video(self):
        """
        Load video frames from the specified video file.
        Converts each frame to RGB format and stores it in the frames list.
        """
        cap = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame from BGR (OpenCV format) to RGB (Qt format)
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    def init_ui(self):
        """
        Initialize the user interface components including buttons, sliders, and layout.
        """
        # Create and configure buttons for navigation and playback
        self.next_frame_button = QtWidgets.QPushButton("Next Frame")
        self.next_frame_button.clicked.connect(self.on_click_next_frame)

        self.prev_frame_button = QtWidgets.QPushButton("Previous Frame")
        self.prev_frame_button.clicked.connect(self.on_click_prev_frame)

        self.jump_forward_button = QtWidgets.QPushButton("Jump Forward 60 Frames")
        self.jump_forward_button.clicked.connect(self.on_click_jump_forward)

        self.jump_backward_button = QtWidgets.QPushButton("Jump Backward 60 Frames")
        self.jump_backward_button.clicked.connect(self.on_click_jump_backward)

        self.play_pause_button = QtWidgets.QPushButton("Play")
        self.play_pause_button.clicked.connect(self.on_click_play_pause)

        # Configure image label to display video frames
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.update_frame_display()

        # Configure slider for navigating through frames
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(self.frames) - 1)
        self.frame_slider.sliderMoved.connect(self.on_move)

        # Arrange widgets in a vertical layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.next_frame_button)
        self.layout.addWidget(self.prev_frame_button)
        self.layout.addWidget(self.jump_forward_button)
        self.layout.addWidget(self.jump_backward_button)
        self.layout.addWidget(self.play_pause_button)
        self.layout.addWidget(self.frame_slider)

    @QtCore.Slot()
    def on_click_next_frame(self):
        """
        Handle the 'Next Frame' button click event.
        Advances to the next frame if not currently playing.
        """
        if not self.playing:
            self.current_frame = min(self.current_frame + 1, len(self.frames) - 1)
            self.update_frame()

    @QtCore.Slot()
    def on_click_prev_frame(self):
        """
        Handle the 'Previous Frame' button click event.
        Moves to the previous frame if not currently playing.
        """
        if not self.playing:
            self.current_frame = max(self.current_frame - 1, 0)
            self.update_frame()

    @QtCore.Slot()
    def on_click_jump_forward(self):
        """
        Handle the 'Jump Forward 60 Frames' button click event.
        Jumps forward by 60 frames if not currently playing.
        """
        if not self.playing:
            self.current_frame = min(self.current_frame + 60, len(self.frames) - 1)
            self.update_frame()

    @QtCore.Slot()
    def on_click_jump_backward(self):
        """
        Handle the 'Jump Backward 60 Frames' button click event.
        Jumps backward by 60 frames if not currently playing.
        """
        if not self.playing:
            self.current_frame = max(self.current_frame - 60, 0)
            self.update_frame()

    @QtCore.Slot()
    def on_click_play_pause(self):
        """
        Handle the 'Play/Pause' button click event.
        Toggles between playing and pausing the video playback.
        """
        if self.playing:
            self.timer.stop()
            self.play_pause_button.setText("Play")
        else:
            self.timer.start(30)  # Set playback speed (30 ms delay for ~33 FPS)
            self.play_pause_button.setText("Pause")
        self.playing = not self.playing

    @QtCore.Slot()
    def update_frame(self):
        """
        Update the displayed frame.
        Processes the current frame using the motion detector and updates the frame slider position.
        """
        if 0 <= self.current_frame < len(self.frames):
            frame = self.frames[self.current_frame]
            self.motion_detector.update_tracking(frame)
            self.update_frame_display()
            self.frame_slider.setValue(self.current_frame)
            self.current_frame = min(self.current_frame + 1, len(self.frames) - 1)

    @QtCore.Slot()
    def update_frame_display(self):
        """
        Update the image display with the current frame.
        """
        frame = self.frames[self.current_frame]
        self.display_frame(frame)

    def display_frame(self, frame):
        """
        Convert the frame to a Qt image format and display it.

        Parameters:
        - frame: The video frame to display.
        """
        h, w, c = frame.shape
        img = QtGui.QImage(frame, w, h, w * c, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))

    @QtCore.Slot()
    def on_move(self):
        """
        Handle slider movement to jump to a specific frame.
        """
        self.current_frame = self.frame_slider.value()
        self.update_frame_display()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for loading video with PySide2.")
    parser.add_argument("video_path", metavar='PATH_TO_VIDEO', type=str, help="Path to the video file.")
    args = parser.parse_args()

    # Create an instance of MotionDetector with specified parameters
    motion_detector = MotionDetector(
        frame_hysteresis=3,
        motion_threshold=25,
        distance_threshold=50,
        skip_frames=1,
        max_objects=10
    )

    app = QtWidgets.QApplication([])

    # Create and show the main application window
    main_window = MainWindow(args.video_path, motion_detector)
    main_window.resize(800, 600)
    main_window.show()

    sys.exit(app.exec_())
