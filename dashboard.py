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
            self.resize(800, 480)  # RPi 7-inch display
            
        self.init_ui()
        self.start_threads()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main vertical layout to hold horizontal content + status bar
        main_vertical_layout = QVBoxLayout(central_widget)
        main_vertical_layout.setContentsMargins(8, 8, 8, 0)
        main_vertical_layout.setSpacing(8)
        
        # Horizontal layout for content (video left, controls right)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(8)

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
        self.video_label.setMaximumHeight(500)  # Prevent excessive growth
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.roi_selected.connect(self.on_roi_drawn)
        left_layout.addWidget(self.video_label, 3)

        # Telemetry Graph — compact
        graph_card = QFrame()
        graph_card.setObjectName("Card")
        graph_layout = QVBoxLayout(graph_card)
        graph_layout.setContentsMargins(3, 3, 3, 3)
        graph_layout.setSpacing(2)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#f7f9fb')
        self.plot_widget.showAxis('bottom')
        self.plot_widget.setLabel('bottom', "Time", units="s", **{'font-size': '7pt'})
        self.plot_widget.hideAxis('left')
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.getPlotItem().setContentsMargins(1, 1, 1, 0)  # No bottom margin for axis at edge
        self.plot_widget.getPlotItem().setLimits(yMin=0)  # No negative values
        self.motion_data = collections.deque(maxlen=150)

        self.curve = self.plot_widget.plot(name="Motion", pen=pg.mkPen(color='#005db5', width=2))
        graph_layout.addWidget(self.plot_widget, 1)

        graph_card.setFixedHeight(110)
        left_layout.addStretch()  # Push graph to bottom
        left_layout.addWidget(graph_card)

        left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout.addWidget(left_widget, 3)

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
            hw_layout.addWidget(QLabel("Temp Control", font=QFont("Segoe UI", 10, QFont.Weight.Bold)))

            temp_header = QHBoxLayout()
            temp_header.setSpacing(2)
            temp_header.addWidget(QLabel("Core:", font=QFont("Segoe UI", 9, QFont.Weight.Normal)))
            self.lbl_target = QLabel("37.5 °C", font=QFont("Segoe UI", 9, QFont.Weight.Bold))
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
                    width: 14px;
                    height: 14px;
                    margin: -5px 0;
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

        alarm_layout.addWidget(QLabel("BPM Control", font=QFont("Segoe UI", 10, QFont.Weight.Bold)))

        filter_header = QHBoxLayout()
        filter_header.setSpacing(2)
        filter_header.addWidget(QLabel("Range:", font=QFont("Segoe UI", 9, QFont.Weight.Normal)))
        self.lbl_bpm_range = QLabel("50-80", font=QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_bpm_range.setStyleSheet("color: #005db5; font-weight: bold;")
        filter_header.addWidget(self.lbl_bpm_range)
        alarm_layout.addLayout(filter_header)

        self.bpm_slider = RangeSlider(minimum=20, maximum=100)
        self.bpm_slider.setValues(50, 80)
        self.bpm_slider.valueChanged.connect(self.on_bpm_range_changed)
        self.bpm_slider.setMinimumHeight(26)
        alarm_layout.addWidget(self.bpm_slider)

        self.cb_alarm = QCheckBox("BPM alarm")
        self.cb_alarm.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                color: #005db5;
                font-size: 10px;
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
                font-size: 10px;
            }
        """)
        self.toggle_yolo.toggled.connect(self.on_yolo_toggled)
        alarm_layout.addWidget(self.toggle_yolo)

        self.btn_manual_roi = QPushButton("Manual ROI")
        self.btn_manual_roi.setFixedSize(100, 26)
        btn_stylesheet = """QPushButton {
            background-color: #005db5;
            color: white;
            font-weight: bold;
            font-size: 10px;
            border-radius: 4px;
            padding: 0px;
            border: none;
        }
        QPushButton:hover {
            background-color: #004a8f;
        }
        QPushButton:pressed {
            background-color: #003d75;
        }"""
        self.btn_manual_roi.setStyleSheet(btn_stylesheet)
        self.btn_manual_roi.clicked.connect(self.activate_drawing_mode)
        alarm_layout.addWidget(self.btn_manual_roi)

        self.btn_record = QPushButton("Record")
        self.btn_record.setFixedSize(100, 26)
        self.btn_record.setStyleSheet(btn_stylesheet)
        self.btn_record.clicked.connect(self.toggle_recording)
        alarm_layout.addWidget(self.btn_record)

        self.alarm_min_bpm = 50
        self.alarm_max_bpm = 80
        self.alarm_trigger_start = None

        right_col.addWidget(alarm_card, 1)

        # Telemetry Info Card
        info_card = QFrame()
        info_card.setObjectName("Card")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(6, 6, 6, 6)
        info_layout.setSpacing(4)

        info_layout.addWidget(QLabel("Telemetry", font=QFont("Segoe UI", 10, QFont.Weight.Bold)))

        # Status line
        status_line = QHBoxLayout()
        status_line.setSpacing(4)
        status_line.setContentsMargins(0, 0, 0, 0)
        status_line.addWidget(QLabel("Status:", font=QFont("Segoe UI", 9, QFont.Weight.Normal)))
        self.lbl_status = QLabel("System Initializing...")
        self.lbl_status.setFont(QFont("Segoe UI", 9, QFont.Weight.Normal))
        self.lbl_status.setStyleSheet("color: #666;")
        status_line.addWidget(self.lbl_status)
        status_line.addStretch()
        info_layout.addLayout(status_line)

        # BPM line
        bpm_line = QHBoxLayout()
        bpm_line.setSpacing(4)
        bpm_line.setContentsMargins(0, 0, 0, 0)
        bpm_line.addWidget(QLabel("Breathing:", font=QFont("Segoe UI", 9, QFont.Weight.Normal)))
        self.lbl_bpm = QLabel("-- BPM")
        self.lbl_bpm.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_bpm.setStyleSheet("color: #005db5")
        bpm_line.addWidget(self.lbl_bpm)
        bpm_line.addStretch()
        info_layout.addLayout(bpm_line)

        if not getattr(self.args, 'breath_only', False):
            # Core temp line
            core_line = QHBoxLayout()
            core_line.setSpacing(4)
            core_line.setContentsMargins(0, 0, 0, 0)
            core_line.addWidget(QLabel("Core:", font=QFont("Segoe UI", 9, QFont.Weight.Normal)))
            self.lbl_mouse_temp = QLabel("--.- °C")
            self.lbl_mouse_temp.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            self.lbl_mouse_temp.setStyleSheet("color: #005db5")
            core_line.addWidget(self.lbl_mouse_temp)
            core_line.addStretch()
            info_layout.addLayout(core_line)

            # Bed temp line
            bed_line = QHBoxLayout()
            bed_line.setSpacing(4)
            bed_line.setContentsMargins(0, 0, 0, 0)
            bed_line.addWidget(QLabel("Bed:", font=QFont("Segoe UI", 9, QFont.Weight.Normal)))
            self.lbl_bed_temp = QLabel("--.- °C")
            self.lbl_bed_temp.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            self.lbl_bed_temp.setStyleSheet("color: #005db5")
            bed_line.addWidget(self.lbl_bed_temp)
            bed_line.addStretch()
            info_layout.addLayout(bed_line)

        info_card.setFixedHeight(140)
        right_col.addWidget(info_card)

        content_layout.addWidget(right_widget, 1)
        
        # Add content to main vertical layout
        main_vertical_layout.addLayout(content_layout, 1)

        self.apply_stylesheet()

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_record.setText("Stop")
            # Filename only - full path handled by camera worker
            filename = f"record_{int(time.time())}.mp4"
            self.cam_worker.start_recording(filename)
        else:
            self.is_recording = False
            self.btn_record.setText("Record")
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
        try:
            if hasattr(self, 'hw_worker'):
                temp = value / 10.0
                self.lbl_target.setText(f"{temp:.1f} °C")
                self.hw_worker.set_target(temp)
        except RuntimeError:
            pass

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
        try:
            if not hasattr(self, 'video_label') or self.video_label is None:
                return
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)
        except RuntimeError:
            # Widget has been deleted
            pass

    def update_tracking_status(self, text: str):
        try:
            if not hasattr(self, 'lbl_status') or self.lbl_status is None:
                return
            self.lbl_status.setText(text)
        except RuntimeError:
            pass

    def update_bpm(self, bpm):
        try:
            if not hasattr(self, 'lbl_bpm') or self.lbl_bpm is None:
                return
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
        except RuntimeError:
            pass
    


    def update_graph(self, motion):
        try:
            if not hasattr(self, 'plot_widget') or self.plot_widget is None:
                return
            if not hasattr(self, 'curve') or self.curve is None:
                return
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
        except RuntimeError:
            pass


    def update_temps(self, mouse, bed, pwm):
        try:
            if not hasattr(self, 'lbl_mouse_temp') or self.lbl_mouse_temp is None:
                return
            if not hasattr(self, 'lbl_bed_temp') or self.lbl_bed_temp is None:
                return
            self.lbl_mouse_temp.setText(f"{mouse:.1f} °C")
            self.lbl_bed_temp.setText(f"{bed:.1f} °C")
        except RuntimeError:
            pass

    def closeEvent(self, event):
        try:
            # Disconnect all signals
            if hasattr(self, 'cam_worker') and self.cam_worker:
                try:
                    self.cam_worker.frame_ready.disconnect()
                    self.cam_worker.bpm_updated.disconnect()
                    self.cam_worker.status_updated.disconnect()
                    self.cam_worker.motion_updated.disconnect()
                except TypeError:
                    pass
            
            if hasattr(self, 'hw_worker') and self.hw_worker:
                try:
                    self.hw_worker.temps_updated.disconnect()
                except TypeError:
                    pass
            
            # Stop threads
            if hasattr(self, 'cam_worker') and self.cam_worker:
                self.cam_worker.running = False
                self.cam_worker.wait(500)
            
            if hasattr(self, 'hw_worker') and self.hw_worker:
                self.hw_worker.running = False
                self.hw_worker.wait(500)
        except Exception as e:
            print(f"Error during closeEvent: {e}")
        
        event.accept()
