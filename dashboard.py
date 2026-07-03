import os
import platform
import time
import collections
import subprocess

if platform.system() == "Windows":
    import winsound

# Check if running on Raspberry Pi Raspbian
IS_RASPBERRY_PI = False
GPIO_AVAILABLE = False
try:
    # Check for RPi by looking at /proc/device-tree/model or architecture
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            if 'Raspberry Pi' in model:
                IS_RASPBERRY_PI = True
    
    if IS_RASPBERRY_PI:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.OUT)
        GPIO.output(17, GPIO.LOW)  # Ensure buzzer starts off
        GPIO_AVAILABLE = True
except (ImportError, RuntimeError, FileNotFoundError):
    # GPIO library not available or not on RPi
    GPIO_AVAILABLE = False

import numpy as np

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFrame, QSlider, QCheckBox, QSizePolicy)
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
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # --- LEFT SIDE: Video + Graph ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.is_recording = False

        # Video feed
        self.video_label = InteractiveVideoLabel()
        self.video_label.setStyleSheet("border-radius: 8px;")
        self.video_label.setMinimumSize(480, 300)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.roi_selected.connect(self.on_roi_drawn)
        left_layout.addWidget(self.video_label, 3)

        # Telemetry Graph — compact
        graph_card = QFrame()
        graph_card.setObjectName("Card")
        graph_layout = QVBoxLayout(graph_card)
        graph_layout.setContentsMargins(6, 6, 6, 6)
        graph_layout.setSpacing(4)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)
        header_layout.addWidget(QLabel("Breathing:", font=QFont("Segoe UI", 10, QFont.Weight.Normal)))

        self.lbl_bpm = QLabel("-- BPM")
        self.lbl_bpm.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_bpm.setStyleSheet("color: #005db5")
        header_layout.addWidget(self.lbl_bpm)
        header_layout.addSpacing(12)
        
        if not getattr(self.args, 'breath_only', False):
            header_layout.addWidget(QLabel("Core:", font=QFont("Segoe UI", 9, QFont.Weight.Normal)))
            self.lbl_mouse_temp = QLabel("--.- °C")
            self.lbl_mouse_temp.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            self.lbl_mouse_temp.setStyleSheet("color: #005db5")
            header_layout.addWidget(self.lbl_mouse_temp)
            header_layout.addSpacing(8)
            
            header_layout.addWidget(QLabel("Bed:", font=QFont("Segoe UI", 9, QFont.Weight.Normal)))
            self.lbl_bed_temp = QLabel("--.- °C")
            self.lbl_bed_temp.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            self.lbl_bed_temp.setStyleSheet("color: #005db5")
            header_layout.addWidget(self.lbl_bed_temp)
        
        header_layout.addStretch()
        graph_layout.addLayout(header_layout)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#f7f9fb')
        self.plot_widget.addLegend(offset=(10, 10))
        self.plot_widget.showAxis('bottom')
        self.plot_widget.setLabel('bottom', "Time", units="s", **{'font-size': '8pt'})
        self.plot_widget.hideAxis('left')
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.motion_data = collections.deque(maxlen=150)

        self.curve = self.plot_widget.plot(name="Motion", pen=pg.mkPen(color='#005db5', width=2))
        graph_layout.addWidget(self.plot_widget)

        graph_card.setFixedHeight(70)
        left_layout.addWidget(graph_card)

        main_layout.addWidget(left_widget, 4)

        # --- RIGHT SIDE: Controls (Narrow) ---
        right_widget = QWidget()
        right_col = QVBoxLayout(right_widget)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(6)

        # Hardware / PID Control Card (compact)
        if not getattr(self.args, 'breath_only', False):
            self.hw_card = QFrame()
            self.hw_card.setObjectName("Card")
            hw_layout = QVBoxLayout(self.hw_card)
            hw_layout.setContentsMargins(6, 6, 6, 6)
            hw_layout.setSpacing(4)
            hw_layout.addWidget(QLabel("Temp Control", font=QFont("Segoe UI", 9, QFont.Weight.Bold)))

            temp_header = QHBoxLayout()
            temp_header.setSpacing(2)
            temp_header.addWidget(QLabel("Core:", font=QFont("Segoe UI", 8, QFont.Weight.Normal)))
            self.lbl_target = QLabel("37.5 °C", font=QFont("Segoe UI", 8, QFont.Weight.Bold))
            self.lbl_target.setStyleSheet("color: #005db5; font-weight: bold;")
            temp_header.addWidget(self.lbl_target)
            hw_layout.addLayout(temp_header)

            self.slider_temp = QSlider(Qt.Orientation.Horizontal)
            self.slider_temp.setRange(300, 400)
            self.slider_temp.setValue(375)
            self.slider_temp.setStyleSheet("""
                QSlider::groove:horizontal {
                    background: #e1e9ee;
                    height: 3px;
                    border-radius: 1px;
                }
                QSlider::sub-page:horizontal {
                    background: #005db5;
                    border-radius: 1px;
                }
                QSlider::add-page:horizontal {
                    background: #e1e9ee;
                    border-radius: 1px;
                }
                QSlider::handle:horizontal {
                    background: white;
                    border: 1px solid #ccc;
                    width: 12px;
                    height: 12px;
                    margin: -4px 0;
                    border-radius: 6px;
                }
            """)
            self.slider_temp.valueChanged.connect(self.on_target_changed)
            hw_layout.addWidget(self.slider_temp)
            right_col.addWidget(self.hw_card)

        # BPM Control Card (compact)
        alarm_card = QFrame()
        alarm_card.setObjectName("Card")
        alarm_layout = QVBoxLayout(alarm_card)
        alarm_layout.setContentsMargins(6, 6, 6, 6)
        alarm_layout.setSpacing(4)

        alarm_layout.addWidget(QLabel("BPM Control", font=QFont("Segoe UI", 9, QFont.Weight.Bold)))

        filter_header = QHBoxLayout()
        filter_header.setSpacing(2)
        filter_header.addWidget(QLabel("Range:", font=QFont("Segoe UI", 8, QFont.Weight.Normal)))
        self.lbl_bpm_range = QLabel("50-80", font=QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.lbl_bpm_range.setStyleSheet("color: #005db5; font-weight: bold;")
        filter_header.addWidget(self.lbl_bpm_range)
        alarm_layout.addLayout(filter_header)

        self.bpm_slider = RangeSlider(minimum=20, maximum=100)
        self.bpm_slider.setValues(50, 80)
        self.bpm_slider.valueChanged.connect(self.on_bpm_range_changed)
        self.bpm_slider.setMinimumHeight(24)
        alarm_layout.addWidget(self.bpm_slider)

        self.cb_alarm = QCheckBox("BPM alarm")
        self.cb_alarm.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                color: #005db5;
                font-size: 9px;
            }
        """)
        self.cb_alarm.setChecked(True)
        alarm_layout.addWidget(self.cb_alarm)

        # ROI controls (compact)
        self.toggle_yolo = QCheckBox("Auto ROI")
        self.toggle_yolo.setChecked(True)
        self.toggle_yolo.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                color: #005db5;
                font-size: 9px;
            }
        """)
        self.toggle_yolo.toggled.connect(self.on_yolo_toggled)
        alarm_layout.addWidget(self.toggle_yolo)

        self.btn_manual_roi = QPushButton("Manual ROI")
        self.btn_manual_roi.setFixedHeight(24)
        self.btn_manual_roi.setStyleSheet("background-color: #005db5; color: white; font-weight: bold; font-size: 8px; border-radius: 4px; padding: 2px;")
        self.btn_manual_roi.clicked.connect(self.activate_drawing_mode)
        alarm_layout.addWidget(self.btn_manual_roi)

        self.btn_record = QPushButton("RECORD")
        self.btn_record.setFixedHeight(24)
        self.btn_record.setStyleSheet("background-color: #005db5; color: white; font-weight: bold; font-size: 8px; border-radius: 4px; padding: 2px;")
        self.btn_record.clicked.connect(self.toggle_recording)
        alarm_layout.addWidget(self.btn_record)

        self.alarm_min_bpm = 50
        self.alarm_max_bpm = 80
        self.alarm_trigger_start = None

        right_col.addWidget(alarm_card)
        right_col.addStretch()

        main_layout.addWidget(right_widget, 1)

        # Info bar — full width
        self.lbl_status = QLabel("System Initializing...")
        self.lbl_status.setObjectName("InfoBar")
        self.lbl_status.setFixedHeight(28)
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.lbl_status.setStyleSheet("font-size: 9px;")
        
        # Create a bottom bar container
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        bottom_layout.addWidget(self.lbl_status)
        
        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_layout)
        
        # Add bottom to main layout
        main_widget = QWidget()
        main_widget_layout = QVBoxLayout(main_widget)
        main_widget_layout.setContentsMargins(0, 0, 0, 0)
        main_widget_layout.setSpacing(0)
        main_widget_layout.addLayout(main_layout, 1)
        main_widget_layout.addWidget(bottom_widget)
        
        self.setCentralWidget(main_widget)

        self.apply_stylesheet()

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_record.setText("STOP RECORDING")
            self.btn_record.setStyleSheet("background-color: #001f4d; color: white; font-weight: bold; border-radius: 6px;")
            # Filename only - full path handled by camera worker
            filename = f"record_{int(time.time())}.mp4"
            self.cam_worker.start_recording(filename)
        else:
            self.is_recording = False
            self.btn_record.setText("RECORD SESSION")
            self.btn_record.setStyleSheet("background-color: #005db5; color: white; font-weight: bold; border-radius: 6px;")
            self.cam_worker.stop_recording()

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f7f9fb; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
            #Card { background-color: #ffffff; border: 1px solid #e1e9ee; border-radius: 12px; padding: 10px; }
            #InfoBar { background-color: #1a2530; color: #d0dce5; font-family: 'Consolas', monospace; font-size: 12px; padding-left: 12px; padding-right: 12px; }
            QLabel { color: #2a3439; }
            QCheckBox::indicator:unchecked:hover {
                background-color: #e8f0f7;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #004a8f;
            }
        """)

    def start_threads(self):
        # Camera Thread
        self.cam_worker = CameraWorker(self.args.cameraIndex, self.args.videoPath, self.args.test_mode)
        # Note: cam_worker uses fixed physiological limits (30-300 BPM) for signal processing
        # alarm_min_bpm and alarm_max_bpm are only for triggering alarms
        self.cam_worker.frame_ready.connect(self.update_video)
        self.cam_worker.bpm_updated.connect(self.update_bpm)
        self.cam_worker.status_updated.connect(self.update_tracking_status)
        self.cam_worker.motion_updated.connect(self.update_graph)
        self.cam_worker.start()

        # Hardware Thread
        if not getattr(self.args, 'breath_only', False):
            self.hw_worker = HardwareWorker(self.args.test_mode)
            self.hw_worker.temps_updated.connect(self.update_temps)
            self.hw_worker.start()

    def on_target_changed(self, value):
        temp = value / 10.0
        self.lbl_target.setText(f"{temp:.1f} °C")
        self.hw_worker.set_target(temp)

    def on_bpm_range_changed(self, min_val, max_val):
        self.lbl_bpm_range.setText(f"{min_val} - {max_val} BPM")
        self.alarm_min_bpm = float(min_val)
        self.alarm_max_bpm = float(max_val)
        # Note: Alarm range only affects alarm triggering, not BPM calculation

    def trigger_alarm(self):
        import threading
        current_time = time.time()
        if current_time - getattr(self, 'last_beep_time', 0) > 1.0:
            self.last_beep_time = current_time
            def play_beep():
                if GPIO_AVAILABLE and IS_RASPBERRY_PI:
                    # Raspberry Pi: pulse GPIO 17 buzzer (100ms pulse)
                    try:
                        import RPi.GPIO as GPIO
                        GPIO.output(17, GPIO.HIGH)
                        time.sleep(0.2)  # 100ms pulse
                        GPIO.output(17, GPIO.LOW)
                    except Exception as e:
                        print(f"GPIO error: {e}")
                elif platform.system() == "Windows":
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                else:
                    os.system('printf "\a" > /dev/console 2>/dev/null || (speaker-test -t sine -f 1000 -l 1 & sleep 0.3 ; kill -9 $!) > /dev/null 2>&1')
            threading.Thread(target=play_beep, daemon=True).start()

    def on_yolo_toggled(self, checked):
        self.cam_worker.set_use_yolo(checked)
        # When YOLO is re-enabled, reset and restart detection
        if checked:
            self.cam_worker.reset_roi()
        
    def activate_drawing_mode(self):
        # Disable auto-ROI detection
        self.toggle_yolo.setChecked(False)
        # Change button to dark blue (active state)
        self.btn_manual_roi.setStyleSheet("background-color: #001f4d; color: white; font-weight: bold; border-radius: 6px;")
        # Enable selection on the video label
        self.video_label.enable_selection()
        
    def on_roi_drawn(self, nx, ny, nw, nh):
        # Change button back to normal blue
        self.btn_manual_roi.setStyleSheet("background-color: #005db5; color: white; font-weight: bold; border-radius: 6px;")
        if hasattr(self, 'cam_worker'):
            self.cam_worker.apply_manual_roi(nx, ny, nw, nh)

    def update_video(self, q_img):
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def update_tracking_status(self, text: str):
        self.lbl_status.setText(text)

    def update_bpm(self, bpm):
        import math
        if math.isnan(bpm):
            self.lbl_bpm.setText("--")
            return
        self.lbl_bpm.setText(f"{bpm:.1f}")
        
        if self.cb_alarm.isChecked() and (bpm < self.alarm_min_bpm or bpm > self.alarm_max_bpm) and bpm > 0:
            if getattr(self, 'alarm_trigger_start', None) is None:
                self.alarm_trigger_start = time.time()
            elif time.time() - self.alarm_trigger_start > 5.0:
                self.trigger_alarm()
        else:
            self.alarm_trigger_start = None
    


    def update_graph(self, motion):
        self.motion_data.append(motion)
        
        num_points = len(self.motion_data)
        x_data = np.linspace(-num_points / 30.0, 0.0, num_points)
        
        self.curve.setData(x_data, list(self.motion_data))

        if num_points > 10:
            recent_motion = list(self.motion_data)
            
            min_y = min(recent_motion)
            max_y = max(recent_motion)
            
            padding = (max_y - min_y) * 0.1
            if padding < 1e-6: 
                padding = 0.1
                
            self.plot_widget.setYRange(min_y - padding, max_y + padding)


    def update_temps(self, mouse, bed, pwm):
        self.lbl_mouse_temp.setText(f"{mouse:.1f} °C")
        self.lbl_bed_temp.setText(f"{bed:.1f} °C")

    def closeEvent(self, event):
        try:
            self.cam_worker.frame_ready.disconnect()
            self.cam_worker.bpm_updated.disconnect()
            self.cam_worker.status_updated.disconnect()
            self.cam_worker.motion_updated.disconnect()
            if hasattr(self, 'hw_worker'):
                self.hw_worker.temps_updated.disconnect()
        except TypeError:
            pass 

        self.cam_worker.running = False
        if hasattr(self, 'hw_worker'):
            self.hw_worker.running = False
        
        self.cam_worker.wait(500)
        if hasattr(self, 'hw_worker'):
            self.hw_worker.wait(500)
        
        event.accept()
