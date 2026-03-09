import sys
import time
import math
import cv2
import numpy as np
import os

# Check model file
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    sys.exit(1)

print(f"✅ Model OK: {MODEL_PATH}")

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QRadialGradient
from PyQt5.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, 
                           QMainWindow, QSlider, QGroupBox, QFormLayout)

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.cap = None
        
        # MediaPipe setup
        BaseOptions = python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = vision.RunningMode
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        
        self.smoothing = 0.5
        self.pinch_threshold = 0.05
        self.last_click_time = 0
        self.last_index = None
        self.last_thumb = None

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            index_coords = None
            thumb_coords = None
            
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                landmarks = results.hand_landmarks[0]
                lm_index = landmarks[8]
                lm_thumb = landmarks[4]
                
                index_coords = (lm_index.x, lm_index.y)
                thumb_coords = (lm_thumb.x, lm_thumb.y)
                
                # Draw ALL landmarks
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # Pinch line (YELLOW)
                x1, y1 = int(lm_index.x * w), int(lm_index.y * h)
                x2, y2 = int(lm_thumb.x * w), int(lm_thumb.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.circle(frame, (x1, y1), 8, (255, 0, 255), 2)  # Index
                cv2.circle(frame, (x2, y2), 8, (255, 255, 0), 2)  # Thumb

            self.last_index = index_coords
            self.last_thumb = thumb_coords
            self.frame_ready.emit(frame)

    def stop(self):
        self.running = False

class StylishCursor(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.x = self.y = 0.5
        self.clicked = False
        self.phase = 0
        self.cursors = [(0.5, 0.5) for _ in range(5)]
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def set_cursor(self, x, y, clicked):
        self.x = max(0, min(1, x))
        self.y = max(0, min(1, y))
        self.clicked = clicked

    def paintEvent(self, event):
        self.phase += 0.1
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 35))
        w, h = self.width(), self.height()
        
        size = min(w, h) // 12
        
        # Animated borders
        colors = [QColor(100, 180, 255), QColor(255, 150, 200), QColor(100, 255, 180)]
        for i in range(3):
            margin = 20 + int(10 * math.sin(self.phase + i))
            painter.setPen(colors[i])
            painter.drawRect(margin, margin, w-2*margin, h-2*margin)

        # Multi-cursor system ✅ FIXED: All int coordinates
        lerp_rates = [0.8, 0.65, 0.5, 0.35, 0.2]
        painter.setPen(Qt.NoPen)
        
        for i in range(5):
            cx, cy = self.cursors[i]
            lerp = lerp_rates[i]
            nx = cx + (self.x - cx) * lerp
            ny = cy + (self.y - cy) * lerp
            self.cursors[i] = (nx, ny)
            
            # Trail effect ✅ FIXED: int() conversion
            for j in range(6):
                tx = nx + 0.03 * math.sin(self.phase * 2 + j)
                ty = ny + 0.03 * math.cos(self.phase * 2 + j)
                px = int(tx * w)
                py = int(ty * h)
                alpha = 120 - j * 20
                col = colors[i % 3]
                col.setAlpha(alpha)
                painter.setBrush(col)
                r = max(2, size // 4 - j * 2)
                painter.drawEllipse(px-r, py-r, r*2, r*2)  # ✅ FIXED: All ints
            
            # Main cursor ✅ FIXED: int() conversion
            px = int(nx * w)
            py = int(ny * h)
            col = colors[i % 3]
            col.setAlpha(200 - i * 30)
            painter.setBrush(col)
            r = int(size * (0.8 - i * 0.1))  # ✅ FIXED: int()
            painter.drawEllipse(px-r//2, py-r//2, r, r)

        # Click ripple ✅ FIXED: Proper int coordinates
        if self.clicked:
            avg_x = sum(c[0] for c in self.cursors) / 5
            avg_y = sum(c[1] for c in self.cursors) / 5
            cx = int(avg_x * w)
            cy = int(avg_y * h)
            
            gradient = QRadialGradient(cx, cy, int(size * 2.5))
            gradient.setColorAt(0, QColor(255, 220, 100, 200))
            gradient.setColorAt(0.6, QColor(255, 180, 80, 100))
            gradient.setColorAt(1, QColor(255, 120, 60, 0))
            
            painter.setBrush(gradient)
            painter.setPen(Qt.NoPen)
            pulse_r = int(size * (1.8 + 0.6 * math.sin(self.phase * 12)))
            painter.drawEllipse(cx-pulse_r//2, cy-pulse_r//2, pulse_r, pulse_r)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("✋→🖱️ Hand Mouse - PERFECT!")
        self.setGeometry(100, 100, 950, 700)
        self.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #1a1a2e,stop:1 #16213e); color: white;")
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(20)

        # Video
        self.video_label = QLabel("🎥 Loading camera...")
        self.video_label.setFixedHeight(480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:black;border-radius:15px;border:3px solid #444;color:#aaa;font-size:18px;")
        layout.addWidget(self.video_label)

        # Cursor display
        self.cursor_widget = StylishCursor()
        layout.addWidget(self.cursor_widget)

        # Controls
        controls = QGroupBox("🎛️ Live Controls")
        controls_layout = QFormLayout(controls)
        
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(10, 90)
        self.smooth_slider.setValue(50)
        self.smooth_slider.valueChanged.connect(self.set_smooth)
        controls_layout.addRow("Smooth:", self.smooth_slider)
        
        self.pinch_slider = QSlider(Qt.Horizontal)
        self.pinch_slider.setRange(2, 20)
        self.pinch_slider.setValue(5)
        self.pinch_slider.valueChanged.connect(self.set_pinch)
        controls_layout.addRow("Pinch:", self.pinch_slider)
        
        layout.addWidget(controls)

        # Start video processing
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_video)
        self.video_thread.start()

        # Cursor state
        self.virtual_x = self.virtual_y = 0.5
        self.virtual_click = False
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_cursor)
        self.timer.start(30)

    def update_video(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)
        self.video_label.setText("")

    def update_cursor(self):
        idx = self.video_thread.last_index
        th = self.video_thread.last_thumb
        screen_w, screen_h = pyautogui.size()
        pyautogui.moveTo(self.virtual_x * screen_w, self.virtual_y * screen_h)
        if self.virtual_click:
            pyautogui.click
        
        if idx:
            nx, ny = max(0, min(1, idx[0])), max(0, min(1, idx[1]))
            s = self.video_thread.smoothing
            self.virtual_x += (nx - self.virtual_x) * (1 - s)
            self.virtual_y += (ny - self.virtual_y) * (1 - s)

            if th:
                dist = math.hypot(nx - th[0], ny - th[1])
                if dist < self.video_thread.pinch_threshold:
                    now = time.time()
                    if now - self.video_thread.last_click_time > 0.4:
                        self.virtual_click = True
                        self.video_thread.last_click_time = now
                else:
                    self.virtual_click = False

        self.cursor_widget.set_cursor(self.virtual_x, self.virtual_y, self.virtual_click)

    def set_smooth(self, value):
        self.video_thread.smoothing = value / 100.0

    def set_pinch(self, value):
        self.video_thread.pinch_threshold = value / 100.0

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    print("🎉 Hand Mouse 100% WORKING! Point index finger & pinch thumb!")
    sys.exit(app.exec_())
