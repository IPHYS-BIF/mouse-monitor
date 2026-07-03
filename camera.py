import cv2
import time
import os
import shutil
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from signal_processing import AdvancedBreathEstimator
from vision_tracking import RoiState, autoDetectRoiYolo, updateRoiByTemplate, computeRoiMotion

class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    bpm_updated = Signal(float)
    status_updated = Signal(str)
    motion_updated = Signal(float)

    def __init__(self, camera_index=0, video_path="", test_mode=False):
        super().__init__()
        self.running = True
        self.camera_index = camera_index
        self.video_path = video_path
        self.test_mode = test_mode
        self.min_bpm = 30.0  # Fixed physiological minimum
        self.max_bpm = 300.0  # Fixed physiological maximum
        self.use_yolo = True
        self.manual_roi_pending = None
        self.recording_active = False
        self.recording_filename = ""
        self.video_writer = None
        self._yolo_search_frame = 0
        self._yolo_failed_attempts = 0

    def start_recording(self, filename):
        """Start recording. Creates data/video directory and checks disk space first."""
        # Create data/video directory if it doesn't exist
        video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "video")
        os.makedirs(video_dir, exist_ok=True)
        
        # Check available disk space (require at least 100MB free)
        try:
            stat = shutil.disk_usage(video_dir)
            free_space_mb = stat.free / (1024 * 1024)
            min_space_mb = 100
            if free_space_mb < min_space_mb:
                print(f"Insufficient disk space: {free_space_mb:.1f}MB free, need {min_space_mb}MB")
                self.status_updated.emit(f"ERROR: Only {free_space_mb:.0f}MB free (need {min_space_mb}MB)")
                return
        except Exception as e:
            print(f"Warning: Could not check disk space: {e}")
        
        # Set full path for recording file
        self.recording_filename = os.path.join(video_dir, filename)
        self.recording_active = True
        self.status_updated.emit(f"RECORDING to {filename}")

    def stop_recording(self):
        self.recording_active = False
    
    def set_use_yolo(self, state):
        self.use_yolo = state
    
    def reset_roi(self):
        """Reset ROI and YOLO state to restart detection from scratch."""
        self.manual_roi_pending = None
        self._roi_state_pending = None  # Signal to reset in run loop
        
    def apply_manual_roi(self, nx, ny, nw, nh):
        self.manual_roi_pending = (nx, ny, nw, nh)

    def run(self):
        source = self.video_path if self.video_path else self.camera_index
        capture = cv2.VideoCapture(source)

        isVideoFile = bool(self.video_path)
        videoFps = capture.get(cv2.CAP_PROP_FPS) if isVideoFile else 0.0
        processedFrameCount = 0
        
        method = "correlation"  # or "correlation"
        estimator = AdvancedBreathEstimator(method=method, breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
        roiState = None
        self._roi_state_pending = None
        runStartTime = time.perf_counter()

        while self.running:
            # Check if ROI reset was requested
            if self._roi_state_pending is None and roiState is not None:
                roiState = None
                self._yolo_search_frame = 0
                self._yolo_failed_attempts = 0
                estimator = AdvancedBreathEstimator(method=method, breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
                self.motion_updated.emit(0.0)
                self.bpm_updated.emit(float('nan'))
                self.status_updated.emit("Tracking...")
                self._roi_state_pending = False  # Mark reset as processed
            
            ret, frame = capture.read()
            if not ret:
                if self.video_path: # Loop video for testing
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            processedFrameCount += 1

            if self.recording_active:
                if getattr(self, 'video_writer', None) is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = videoFps if videoFps > 0 else 30.0
                    if fps <= 0 or fps > 120: fps = 30.0
                    self.video_writer = cv2.VideoWriter(self.recording_filename, int(fourcc), float(fps), (frame.shape[1], frame.shape[0]))
                if self.video_writer is not None and self.video_writer.isOpened():
                    self.video_writer.write(frame)
            else:
                if getattr(self, 'video_writer', None) is not None:
                    self.video_writer.release()
                    self.video_writer = None

            if isVideoFile and videoFps > 0:
                nowTime = processedFrameCount / videoFps
            else:
                nowTime = time.perf_counter() - runStartTime

            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # hybrid tracking
            if self.manual_roi_pending is not None:
                nx, ny, nw, nh = self.manual_roi_pending
                self.manual_roi_pending = None
                fh, fw = frame.shape[:2]
                
                x, y = int(nx * fw), int(ny * fh)
                w, h = int(nw * fw), int(nh * fh)
                
                templateGray = grayFrame[y : y + h, x : x + w].copy()
                roiState = RoiState(x=x, y=y, width=w, height=h, templateGray=templateGray)
                estimator = AdvancedBreathEstimator(method=method, breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
                self.motion_updated.emit(0.0)
                self.bpm_updated.emit(float('nan'))
                self.status_updated.emit("Detecting breathing...")
                self.prev_gray_frame = None

            # main tracking logic
            if roiState is None:
                if self.use_yolo:
                    self._yolo_search_frame += 1
                    if self._yolo_search_frame % 10 == 1:  # run YOLO every 10 frames
                        roiState = autoDetectRoiYolo(frame)
                        if roiState is not None:
                            self._yolo_search_frame = 0
                            self._yolo_failed_attempts = 0
                            estimator = AdvancedBreathEstimator(method="correlation", breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
                            self.motion_updated.emit(0.0)
                            self.bpm_updated.emit(float('nan'))
                            self.status_updated.emit("Detecting breathing...")
                            self.prev_gray_frame = None
                        else:
                            self._yolo_failed_attempts += 1
                            if self._yolo_failed_attempts >= 30:
                                self.use_yolo = False
                                self.status_updated.emit(
                                    "MODEL NOT FOUND or no detection — Draw ROI manually")
                            else:
                                self.status_updated.emit(
                                    f"Tracking ... ({self._yolo_failed_attempts}/30)")
                else:
                    self.status_updated.emit("WAITING FOR MANUAL ROI (Click & Drag)")
            else:
                roiState, confidence = updateRoiByTemplate(grayFrame, roiState)
                if confidence <= 0.9:
                    # Mouse has moved from last known position — re-acquire
                    roiState = None
                    self._yolo_search_frame = 0
                    self.prev_gray_frame = None
                    self.bpm_updated.emit(float('nan'))
                    self.status_updated.emit("Mouse retracking...")
                else:
                    bpm, _ = estimator.estimateBreath()
                    
                    if bpm is not None:
                        self._bpm_miss_count = 0
                        self.bpm_updated.emit(bpm)
                        self.status_updated.emit("Mouse ROI stable")
                    else:
                        self._bpm_miss_count = getattr(self, '_bpm_miss_count', 0) + 1
                        # Only clear BPM after 90 consecutive missed estimates (~3s at 30fps)
                        if self._bpm_miss_count > 90:
                            self.bpm_updated.emit(float('nan'))
                            if estimator.has_enough_data():
                                self.status_updated.emit("No breathing detected")
                            else:
                                self.status_updated.emit("Detecting breathing...")
                    
                    motion = computeRoiMotion(grayFrame, roiState)
                    self.prev_gray_frame = grayFrame.copy()

                    self.motion_updated.emit(motion)

                    estimator.addSample(nowTime, motion)

                    # Feed raw ROI patch for the correlation method
                    roiPatch = grayFrame[roiState.y: roiState.y + roiState.height,
                                         roiState.x: roiState.x + roiState.width]
                    estimator.addFrame(nowTime, roiPatch)



                    # Draw Box
                    cv2.rectangle(frame, (roiState.x, roiState.y), 
                                 (roiState.x + roiState.width, roiState.y + roiState.height), 
                                 (181, 93, 0), 2)

            # Convert to UI format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qt_image)
            
            if self.test_mode and self.video_path:
                self.msleep(15) # Prevent video from playing at 1000 FPS

        capture.release()

    def stop(self):
        self.running = False
        if getattr(self, 'video_writer', None) is not None:
            self.video_writer.release()
            self.video_writer = None
        self.wait()
