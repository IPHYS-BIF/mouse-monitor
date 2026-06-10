import cv2
import time
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from signal_processing import BreathEstimator, AdvancedBreathEstimator
from vision_tracking import RoiState, autoDetectRoiYolo, updateRoiByTemplate, computeRoiMotion, computeOpticalFlowMotion

class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    bpm_updated = Signal(tuple)
    status_updated = Signal(str)
    motion_updated = Signal(tuple)

    def __init__(self, camera_index=0, video_path="", test_mode=False):
        super().__init__()
        self.running = True
        self.camera_index = camera_index
        self.video_path = video_path
        self.test_mode = test_mode
        self.min_bpm = 60.0
        self.max_bpm = 240.0
        self.use_yolo = True
        self.manual_roi_pending = None
        self.recording_active = False
        self.recording_filename = ""
        self.video_writer = None

    def start_recording(self, filename):
        self.recording_filename = filename
        self.recording_active = True

    def stop_recording(self):
        self.recording_active = False
    
    def set_use_yolo(self, state):
        self.use_yolo = state
        
    def apply_manual_roi(self, nx, ny, nw, nh):
        self.manual_roi_pending = (nx, ny, nw, nh)

    def run(self):
        source = self.video_path if self.video_path else self.camera_index
        capture = cv2.VideoCapture(source)

        isVideoFile = bool(self.video_path)
        videoFps = capture.get(cv2.CAP_PROP_FPS) if isVideoFile else 0.0
        processedFrameCount = 0
        
        estimator_baseline = BreathEstimator(breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
        estimator_advanced = AdvancedBreathEstimator(breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
        estimator_flow = AdvancedBreathEstimator(breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
        roiState = None
        runStartTime = time.perf_counter()

        while self.running:
            # Sync parameters to estimator
            estimator_baseline.update_limits(self.min_bpm, self.max_bpm)
            estimator_advanced.update_limits(self.min_bpm, self.max_bpm)
            estimator_flow.update_limits(self.min_bpm, self.max_bpm)
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
                estimator_baseline = BreathEstimator(breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
                estimator_advanced = AdvancedBreathEstimator(breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
                estimator_flow = AdvancedBreathEstimator(breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
                self.motion_updated.emit((0.0, 0.0, 0.0))
                self.prev_gray_frame = None

            # main tracking logic
            if roiState is None:
                if self.use_yolo:
                    self.status_updated.emit("SEARCHING FOR TARGET (YOLO)...")
                    roiState = autoDetectRoiYolo(frame)
                    if roiState is not None:
                        estimator_baseline = BreathEstimator(breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
                        estimator_advanced = AdvancedBreathEstimator(breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
                        estimator_flow = AdvancedBreathEstimator(breathMinBpm=self.min_bpm, breathMaxBpm=self.max_bpm)
                        self.motion_updated.emit((0.0, 0.0, 0.0))
                        self.prev_gray_frame = None
                else:
                    self.status_updated.emit("WAITING FOR MANUAL ROI (Click & Drag)")
            else:
                roiState, confidence = updateRoiByTemplate(grayFrame, roiState)
                if confidence < 0.55:
                    roiState = None
                    self.prev_gray_frame = None
                else:
                    self.status_updated.emit(f"TRACKING (Conf: {confidence:.2f})")
                    motion_base = computeRoiMotion(grayFrame, roiState)
                    motion_adv = motion_base
                    
                    if getattr(self, 'prev_gray_frame', None) is not None:
                        motion_flow = computeOpticalFlowMotion(self.prev_gray_frame, grayFrame, roiState)
                    else:
                        motion_flow = 0.0
                        
                    self.prev_gray_frame = grayFrame.copy()
                    
                    self.motion_updated.emit((motion_base, motion_adv, motion_flow))

                    estimator_baseline.addSample(nowTime, motion_base)
                    estimator_advanced.addSample(nowTime, motion_adv)
                    estimator_flow.addSample(nowTime, motion_flow)
                    
                    bpm_base, _ = estimator_baseline.estimateBreath()
                    bpm_adv, _ = estimator_advanced.estimateBreath()
                    bpm_flow, _ = estimator_flow.estimateBreath()
                    
                    if bpm_base is not None or bpm_adv is not None or bpm_flow is not None:
                        b1 = bpm_base if bpm_base is not None else 0.0
                        b2 = bpm_adv if bpm_adv is not None else 0.0
                        b3 = bpm_flow if bpm_flow is not None else 0.0
                        self.bpm_updated.emit((b1, b2, b3))

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
                self.msleep(15) # Prevent video from playing at 1000 FPS

        capture.release()

    def stop(self):
        self.running = False
        if getattr(self, 'video_writer', None) is not None:
            self.video_writer.release()
            self.video_writer = None
        self.wait()
