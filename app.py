import sys
import time
import argparse
import collections
from dataclasses import dataclass
import cv2
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGridLayout, QFrame, QSlider)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
import pyqtgraph as pg

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics is not installed. YOLO detection will fail.")

YOLO_MODEL = None

# ==========================================
# 1. DATA CLASSES & SIGNAL PROCESSING
# ==========================================
@dataclass
class RoiState:
    x: int
    y: int
    width: int
    height: int
    templateGray: np.ndarray

class BreathEstimator:
    def __init__(self, breathMinBpm: float = 60.0, breathMaxBpm: float = 240.0, bufferSeconds: float = 18.0):
        self.breathMinHz = breathMinBpm / 60.0
        self.breathMaxHz = breathMaxBpm / 60.0
        self.bufferSeconds = bufferSeconds
        self.timeSeries = collections.deque()
        self.motionSeries = collections.deque()
        self.smoothedBreathBpm = None

    def addSample(self, sampleTime: float, motionValue: float) -> None:
        self.timeSeries.append(sampleTime)
        self.motionSeries.append(motionValue)
        while self.timeSeries and (sampleTime - self.timeSeries[0]) > self.bufferSeconds:
            self.timeSeries.popleft()
            self.motionSeries.popleft()

    def estimateBreath(self) -> tuple[float | None, float]:
        if len(self.timeSeries) < 90:
            return None, 0.0

        timeValues = np.array(self.timeSeries, dtype=np.float64)
        motionValues = np.array(self.motionSeries, dtype=np.float64)
        duration = timeValues[-1] - timeValues[0]
        
        if duration < max(6.0, 2.5 / self.breathMinHz):
            return None, 0.0

        sampleCount = min(512, int(duration * 30.0))
        uniformTimes = np.linspace(timeValues[0], timeValues[-1], sampleCount)
        uniformSignal = np.interp(uniformTimes, timeValues, motionValues)

        centeredSignal = uniformSignal - np.mean(uniformSignal)
        if np.std(centeredSignal) < 1e-6:
            return None, 0.0

        windowedSignal = centeredSignal * np.hanning(centeredSignal.size)
        samplingRate = sampleCount / duration
        spectrum = np.fft.rfft(windowedSignal)
        power = np.abs(spectrum) ** 2
        frequencies = np.fft.rfftfreq(windowedSignal.size, d=1.0 / samplingRate)

        validMask = (frequencies >= self.breathMinHz) & (frequencies <= self.breathMaxHz)
        if not np.any(validMask):
            return None, 0.0

        validFrequencies = frequencies[validMask]
        validPower = power[validMask]
        peakIndex = int(np.argmax(validPower))
        
        peakFrequency = float(validFrequencies[peakIndex])
        peakPower = float(validPower[peakIndex])
        noiseFloor = float(np.median(validPower) + 1e-9)
        confidence = float(np.clip((peakPower / noiseFloor) / 10.0, 0.0, 1.0))
        breathBpm = peakFrequency * 60.0

        if self.smoothedBreathBpm is None:
            self.smoothedBreathBpm = breathBpm
        else:
            self.smoothedBreathBpm = 0.2 * breathBpm + 0.8 * self.smoothedBreathBpm

        return float(self.smoothedBreathBpm), confidence

# ==========================================
# 2. COMPUTER VISION TRACKING PIPELINE
# ==========================================
def clampRoi(x, y, w, h, fw, fh):
    return max(0, min(x, fw - w)), max(0, min(y, fh - h)), w, h

def autoDetectRoiYolo(frameBgr: np.ndarray) -> RoiState | None:
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO('yolov8n.pt') 
        
    results = YOLO_MODEL(frameBgr, verbose=False)
    if len(results[0].boxes) == 0:
        return None
        
    best_box = None
    best_conf = 0.0
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf > best_conf and conf > 0.45:
            best_conf = conf
            best_box = box

    if best_box is None: return None

    x1, y1, x2, y2 = [int(v) for v in best_box.xyxy[0]]
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * 0.15), int(h * 0.15)
    
    newX, newY = x1 + mx, y1 + my
    newW, newH = max(8, w - 2*mx), max(8, h - 2*my)

    # Note: Extracting Red channel for IR, or grayscale for standard
    gray = frameBgr[:, :, 2].copy() 
    templateGray = gray[newY : newY + newH, newX : newX + newW].copy()
    return RoiState(x=newX, y=newY, width=newW, height=newH, templateGray=templateGray)

def updateRoiByTemplate(grayFrame: np.ndarray, roiState: RoiState, margin: int = 15) -> tuple[RoiState, float]:
    fh, fw = grayFrame.shape[:2]
    sx, sy = max(0, roiState.x - margin), max(0, roiState.y - margin)
    sx2, sy2 = min(fw, roiState.x + roiState.width + margin), min(fh, roiState.y + roiState.height + margin)

    searchRegion = grayFrame[sy:sy2, sx:sx2]
    template = roiState.templateGray

    if searchRegion.shape[0] < template.shape[0] or searchRegion.shape[1] < template.shape[1]:
        return roiState, 0.0

    result = cv2.matchTemplate(searchRegion, template, cv2.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(result)

    newX, newY, _, _ = clampRoi(sx + maxLoc[0], sy + maxLoc[1], roiState.width, roiState.height, fw, fh)
    newPatch = grayFrame[newY : newY + roiState.height, newX : newX + roiState.width]
    updatedTemplate = cv2.addWeighted(template, 0.92, newPatch, 0.08, 0.0)

    return RoiState(x=newX, y=newY, width=roiState.width, height=roiState.height, templateGray=updatedTemplate), maxVal

def computeRoiMotion(grayFrame: np.ndarray, roiState: RoiState) -> float:
    roi = grayFrame[roiState.y : roiState.y + roiState.height, roiState.x : roiState.x + roiState.width]
    roiBlur = cv2.GaussianBlur(roi, (5, 5), 0)
    motionMap = cv2.absdiff(roiBlur, roiState.templateGray)
    return float(np.mean(motionMap))

# ==========================================
# 3. BACKGROUND WORKERS (THREADS)
# ==========================================
class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self.integral, self.prev_error = 0.0, 0.0

    def compute(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return max(0.0, min(100.0, output)) # Constrain PWM between 0 and 100%

class HardwareWorker(QThread):
    temps_updated = pyqtSignal(float, float, float) # mouse_temp, bed_temp, pwm
    
    def __init__(self, test_mode=False):
        super().__init__()
        self.test_mode = test_mode
        self.running = True
        self.target_temp = 37.5
        self.pid = PIDController(kp=5.0, ki=0.1, kd=1.0, setpoint=self.target_temp)
        
        # Sim physics state
        self.sim_mouse_temp = 34.0
        self.sim_bed_temp = 34.0

    def set_target(self, temp):
        self.target_temp = temp
        self.pid.setpoint = temp

    def run(self):
        last_time = time.time()
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            if self.test_mode:
                # 1. Run PID based on Mouse Temp
                pwm = self.pid.compute(self.sim_mouse_temp, dt)
                
                # 2. Simulate thermal physics (Bed heats up based on PWM, loses heat to ambient)
                bed_heat_rate = (pwm / 100.0) * 0.5 
                ambient_loss = (self.sim_bed_temp - 22.0) * 0.01
                self.sim_bed_temp += (bed_heat_rate - ambient_loss) * dt
                
                # 3. Simulate Mouse Temp (Absorbs heat from bed, loses heat to ambient)
                mouse_heat_gain = (self.sim_bed_temp - self.sim_mouse_temp) * 0.05
                mouse_heat_loss = (self.sim_mouse_temp - 22.0) * 0.005
                self.sim_mouse_temp += (mouse_heat_gain - mouse_heat_loss) * dt
                
                self.temps_updated.emit(self.sim_mouse_temp, self.sim_bed_temp, pwm)
            else:
                # REAL HARDWARE LOGIC GOES HERE (I2C read, GPIO PWM write)
                pass 
            
            self.msleep(100) # Run loop at 10Hz

    def stop(self):
        self.running = False
        self.wait()

class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    bpm_updated = pyqtSignal(float)
    status_updated = pyqtSignal(str)
    motion_updated = pyqtSignal(float)

    def __init__(self, camera_index=0, video_path="", test_mode=False):
        super().__init__()
        self.running = True
        self.camera_index = camera_index
        self.video_path = video_path
        self.test_mode = test_mode

    def run(self):
        source = self.video_path if self.video_path else self.camera_index
        capture = cv2.VideoCapture(source)
        
        estimator = BreathEstimator()
        roiState = None
        runStartTime = time.perf_counter()

        while self.running:
            ret, frame = capture.read()
            if not ret:
                if self.video_path: # Loop video for testing
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            nowTime = time.perf_counter() - runStartTime
            grayFrame = frame[:, :, 2].copy() # Red channel for IR
            
            # --- HYBRID TRACKING ---
            if roiState is None:
                self.status_updated.emit("SEARCHING FOR TARGET...")
                roiState = autoDetectRoiYolo(frame)
                if roiState is not None:
                    estimator = BreathEstimator() # Reset math on new target
            else:
                roiState, confidence = updateRoiByTemplate(grayFrame, roiState)
                if confidence < 0.55:
                    roiState = None
                else:
                    self.status_updated.emit(f"TRACKING (Conf: {confidence:.2f})")
                    motion = computeRoiMotion(grayFrame, roiState)
                    self.motion_updated.emit(motion)

                    estimator.addSample(nowTime, motion)
                    bpm, _ = estimator.estimateBreath()
                    if bpm is not None:
                        self.bpm_updated.emit(bpm)

                    # Draw Box
                    cv2.rectangle(frame, (roiState.x, roiState.y), 
                                 (roiState.x + roiState.width, roiState.y + roiState.height), 
                                 (0, 255, 0), 2)

            # Convert to UI format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qt_image)
            
            if self.test_mode and self.video_path:
                self.msleep(30) # Prevent video from playing at 1000 FPS

        capture.release()

    def stop(self):
        self.running = False
        self.wait()

# ==========================================
# 4. MAIN USER INTERFACE
# ==========================================
class MouseTrackerDashboard(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Mouse tracking Dashboard")
        if not args.test_mode:
            self.showFullScreen()
        else:
            self.resize(1280, 800)
            
        self.init_ui()
        self.start_threads()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- CONTENT AREA ---
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        top_bar = QHBoxLayout()
        self.lbl_status = QLabel("System Initializing...")
        self.lbl_status.setFont(QFont("Space Grotesk", 14, QFont.Weight.Bold))
        self.btn_record = QPushButton("RECORD SESSION")
        self.btn_record.setObjectName("RecordButton")
        self.btn_record.setFixedSize(160, 45)
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_record)
        content_layout.addLayout(top_bar)

        # Main Grid
        grid_layout = QGridLayout()
        
        # 1. Video Feed
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: #000; border-radius: 10px;")
        self.video_label.setMinimumSize(640, 480)
        grid_layout.addWidget(self.video_label, 0, 0, 2, 1)

        # 2. Hardware / PID Control Card
        hw_card = QFrame()
        hw_card.setObjectName("Card")
        hw_layout = QVBoxLayout(hw_card)
        hw_layout.addWidget(QLabel("THERMAL REGULATOR (PID)"))
        
        # Readings
        self.lbl_mouse_temp = QLabel("Core: --.- °C")
        self.lbl_bed_temp = QLabel("Bed: --.- °C")
        self.lbl_mouse_temp.setFont(QFont("Inter", 18, QFont.Weight.Bold))
        self.lbl_bed_temp.setFont(QFont("Inter", 14))
        hw_layout.addWidget(self.lbl_mouse_temp)
        hw_layout.addWidget(self.lbl_bed_temp)
        
        # Target Control
        hw_layout.addWidget(QLabel("Target Core Temperature:"))
        self.lbl_target = QLabel("37.5 °C")
        self.lbl_target.setStyleSheet("color: #005db5; font-weight: bold;")
        
        self.slider_temp = QSlider(Qt.Orientation.Horizontal)
        self.slider_temp.setRange(300, 400) # 30.0 to 40.0
        self.slider_temp.setValue(375)
        self.slider_temp.valueChanged.connect(self.on_target_changed)
        
        hw_layout.addWidget(self.lbl_target)
        hw_layout.addWidget(self.slider_temp)
        grid_layout.addWidget(hw_card, 0, 1)

        content_layout.addLayout(grid_layout)

        # 3. Telemetry Graph
        graph_card = QFrame()
        graph_card.setObjectName("Card")
        graph_layout = QVBoxLayout(graph_card)
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Breathing Frequency (BPM)"))
        self.lbl_bpm_value = QLabel("-- BPM")
        self.lbl_bpm_value.setFont(QFont("Inter", 24, QFont.Weight.Bold))
        self.lbl_bpm_value.setStyleSheet("color: #005db5;")
        header_layout.addWidget(self.lbl_bpm_value, alignment=Qt.AlignmentFlag.AlignRight)
        graph_layout.addLayout(header_layout)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#f7f9fb')
        self.motion_data = collections.deque(maxlen=150)

        self.bpm_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#005db5', width=3),
            fillLevel=0,
            fillBrush=pg.mkBrush(0, 93, 181, 40)
        )
        graph_layout.addWidget(self.plot_widget)

        content_layout.addWidget(graph_card)
        main_layout.addWidget(content_area)

        self.apply_stylesheet()

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f7f9fb; font-family: 'Inter', sans-serif; }
            #Card { background-color: #ffffff; border: 1px solid #e1e9ee; border-radius: 12px; padding: 10px; }
            #RecordButton { background-color: #9f403d; color: white; font-weight: bold; border-radius: 6px; }
            QLabel { color: #2a3439; }
        """)

    def start_threads(self):
        # Camera Thread
        self.cam_worker = CameraWorker(self.args.cameraIndex, self.args.videoPath, self.args.test_mode)
        self.cam_worker.frame_ready.connect(self.update_video)
        self.cam_worker.bpm_updated.connect(self.update_bpm)
        self.cam_worker.status_updated.connect(self.lbl_status.setText)
        self.cam_worker.motion_updated.connect(self.update_graph)
        self.cam_worker.start()

        # Hardware Thread
        self.hw_worker = HardwareWorker(self.args.test_mode)
        self.hw_worker.temps_updated.connect(self.update_temps)
        self.hw_worker.start()

    def on_target_changed(self, value):
        temp = value / 10.0
        self.lbl_target.setText(f"{temp:.1f} °C")
        self.hw_worker.set_target(temp)

    def update_video(self, q_img):
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def update_bpm(self, bpm):
        self.lbl_bpm_value.setText(f"{bpm:.1f} BPM")

    def update_graph(self, motion_value):
        self.motion_data.append(motion_value)
        self.bpm_curve.setData(list(self.motion_data))

    def update_temps(self, mouse, bed, pwm):
        self.lbl_mouse_temp.setText(f"Core: {mouse:.1f} °C")
        self.lbl_bed_temp.setText(f"Bed: {bed:.1f} °C (Heater: {pwm:.0f}%)")

    def closeEvent(self, event):
        self.cam_worker.stop()
        self.hw_worker.stop()
        event.accept()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameraIndex", type=int, default=0)
    parser.add_argument("--videoPath", type=str, default="")
    parser.add_argument("--test-mode", action="store_true", help="Run UI on PC with simulated hardware")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MouseTrackerDashboard(args)
    window.show()
    sys.exit(app.exec())