import os
import platform
import time
import collections

if platform.system() == "Windows":
    import winsound

import numpy as np

import numpy as np

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGridLayout, QFrame, QSlider, QCheckBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont, QImage

import pyqtgraph as pg

from ui_components import RangeSlider, InteractiveVideoLabel
from hardware import HardwareWorker
from camera import CameraWorker

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
        self.lbl_status.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.btn_record = QPushButton("RECORD SESSION")
        self.btn_record.setObjectName("RecordButton")
        self.btn_record.setFixedSize(160, 45)
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_record)
        content_layout.addLayout(top_bar)

        # Main Grid
        grid_layout = QGridLayout()
        
        # 1. Video Feed and Controls
        video_col = QVBoxLayout()
        
        # UI Toggles
        roi_controls = QHBoxLayout()
        self.toggle_yolo = QCheckBox("Auto-Detect ROI (YOLO)")
        self.toggle_yolo.setChecked(True)
        self.toggle_yolo.setStyleSheet("font-weight: bold; color: #005db5;")
        self.toggle_yolo.toggled.connect(self.on_yolo_toggled)
        
        self.btn_manual_roi = QPushButton("Draw Manual ROI")
        self.btn_manual_roi.setFixedSize(150, 30)
        self.btn_manual_roi.setEnabled(False) # Disabled by default
        self.btn_manual_roi.clicked.connect(self.activate_drawing_mode)
        
        roi_controls.addWidget(self.toggle_yolo)
        roi_controls.addWidget(self.btn_manual_roi)
        roi_controls.addStretch()
        video_col.addLayout(roi_controls)

        # Interactive Video Label
        self.video_label = InteractiveVideoLabel()
        self.video_label.setStyleSheet("background-color: #000; border-radius: 10px;")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.roi_selected.connect(self.on_roi_drawn)
        
        video_col.addWidget(self.video_label)
        grid_layout.addLayout(video_col, 0, 0, 2, 1)

        # 2. Hardware / PID Control Card
        hw_card = QFrame()
        hw_card.setObjectName("Card")
        hw_layout = QVBoxLayout(hw_card)
        hw_layout.addWidget(QLabel("THERMAL REGULATOR (PID)"))
        
        # Readings
        self.lbl_mouse_temp = QLabel("Core: --.- °C")
        self.lbl_bed_temp = QLabel("Bed: --.- °C")
        self.lbl_mouse_temp.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.lbl_bed_temp.setFont(QFont("Segoe UI", 14))
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

        # 3. Alarm & Filtering Card
        alarm_card = QFrame()
        alarm_card.setObjectName("Card")
        alarm_layout = QVBoxLayout(alarm_card)
        
        alarm_layout.addWidget(QLabel("BPM FILTER & ALARM", font=QFont("Segoe UI", 12, QFont.Weight.Bold)))
        
        filter_header = QHBoxLayout()
        filter_header.addWidget(QLabel("Valid BPM Range:"))
        self.lbl_bpm_range = QLabel("60 - 240")
        self.lbl_bpm_range.setStyleSheet("color: #005db5; font-weight: bold;")
        filter_header.addStretch()
        filter_header.addWidget(self.lbl_bpm_range)
        alarm_layout.addLayout(filter_header)

        self.bpm_slider = RangeSlider(minimum=30, maximum=400)
        self.bpm_slider.setValues(60, 240)
        self.bpm_slider.valueChanged.connect(self.on_bpm_range_changed)
        alarm_layout.addWidget(self.bpm_slider)

        self.cb_alarm = QCheckBox("Enable High BPM Alarm")
        self.cb_alarm.setStyleSheet("font-weight: bold;")
        alarm_layout.addWidget(self.cb_alarm)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Alarm Threshold:"))
        self.lbl_alarm_thresh = QLabel("150 BPM")
        self.lbl_alarm_thresh.setStyleSheet("color: #9f403d; font-weight: bold;")
        threshold_layout.addStretch()
        threshold_layout.addWidget(self.lbl_alarm_thresh)
        alarm_layout.addLayout(threshold_layout)

        self.slider_alarm = QSlider(Qt.Orientation.Horizontal)
        self.slider_alarm.setRange(30, 400)
        self.slider_alarm.setValue(150)
        self.slider_alarm.valueChanged.connect(self.on_alarm_threshold_changed)
        alarm_layout.addWidget(self.slider_alarm)

        grid_layout.addWidget(alarm_card, 1, 1)

        content_layout.addLayout(grid_layout)

        # 3. Telemetry Graph
        graph_card = QFrame()
        graph_card.setObjectName("Card")
        graph_layout = QVBoxLayout(graph_card)
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Breathing Frequency (BPM)"))
        self.lbl_bpm_value = QLabel("-- BPM")
        self.lbl_bpm_value.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.lbl_bpm_value.setStyleSheet("color: #005db5;")
        header_layout.addWidget(self.lbl_bpm_value, alignment=Qt.AlignmentFlag.AlignRight)
        graph_layout.addLayout(header_layout)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#f7f9fb')
        self.plot_widget.showAxis('bottom')
        self.plot_widget.setLabel('bottom', "Time", units="s")
        self.plot_widget.hideAxis('left')
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.motion_data = collections.deque(maxlen=150)
        self.bpm_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#005db5', width=3)
        )
        graph_layout.addWidget(self.plot_widget)

        content_layout.addWidget(graph_card)
        main_layout.addWidget(content_area)

        self.apply_stylesheet()

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f7f9fb; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
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

    def on_bpm_range_changed(self, min_val, max_val):
        self.lbl_bpm_range.setText(f"{min_val} - {max_val}")
        if hasattr(self, 'cam_worker'):
            self.cam_worker.min_bpm = float(min_val)
            self.cam_worker.max_bpm = float(max_val)

    def on_alarm_threshold_changed(self, val):
        self.lbl_alarm_thresh.setText(f"{val} BPM")

    def trigger_alarm(self):
        import threading
        current_time = time.time()
        if current_time - getattr(self, 'last_beep_time', 0) > 1.0:
            self.last_beep_time = current_time
            def play_beep():
                if platform.system() == "Windows":
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                else:
                    os.system('printf "\a" > /dev/console 2>/dev/null || (speaker-test -t sine -f 1000 -l 1 & sleep 0.3 ; kill -9 $!) > /dev/null 2>&1')
            threading.Thread(target=play_beep, daemon=True).start()

    def on_yolo_toggled(self, checked):
        self.cam_worker.set_use_yolo(checked)
        self.btn_manual_roi.setEnabled(not checked)
        
    def activate_drawing_mode(self):
        self.video_label.enable_selection()
        
    def on_roi_drawn(self, nx, ny, nw, nh):
        if hasattr(self, 'cam_worker'):
            self.cam_worker.apply_manual_roi(nx, ny, nw, nh)

    def update_video(self, q_img):
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def update_bpm(self, bpm):
        self.lbl_bpm_value.setText(f"{bpm:.1f} BPM")
        if self.cb_alarm.isChecked() and bpm > self.slider_alarm.value():
            self.trigger_alarm()

    def update_graph(self, motion_value):
        self.motion_data.append(motion_value)
        num_points = len(self.motion_data)
        x_data = np.linspace(-num_points / 30.0, 0.0, num_points)
        self.bpm_curve.setData(x_data, list(self.motion_data))

        if num_points > 10:
            recent_data = list(self.motion_data)
            min_y = min(recent_data)
            max_y = max(recent_data)
            
            padding = (max_y - min_y) * 0.1
            if padding < 1e-6: 
                padding = 0.1
                
            self.plot_widget.setYRange(min_y - padding, max_y + padding)

    def update_temps(self, mouse, bed, pwm):
        self.lbl_mouse_temp.setText(f"Core: {mouse:.1f} °C")
        self.lbl_bed_temp.setText(f"Bed: {bed:.1f} °C (Heater: {pwm:.0f}%)")

    def closeEvent(self, event):
        try:
            self.cam_worker.frame_ready.disconnect()
            self.cam_worker.bpm_updated.disconnect()
            self.cam_worker.status_updated.disconnect()
            self.cam_worker.motion_updated.disconnect()
            self.hw_worker.temps_updated.disconnect()
        except TypeError:
            pass 

        self.cam_worker.running = False
        self.hw_worker.running = False
        
        self.cam_worker.wait(500)
        self.hw_worker.wait(500)
        
        event.accept()
