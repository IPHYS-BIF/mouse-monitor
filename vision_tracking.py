import cv2
import numpy as np
from dataclasses import dataclass

YOLO_MODEL = None

@dataclass
class RoiState:
    x: int
    y: int
    width: int
    height: int
    templateGray: np.ndarray


def clampRoi(x, y, w, h, fw, fh):
    return max(0, min(x, fw - w)), max(0, min(y, fh - h)), w, h

def autoDetectRoiYolo(frameBgr: np.ndarray) -> RoiState | None:
    global YOLO_MODEL
    if YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            YOLO_MODEL = YOLO('yolov8n.pt') 
        except ImportError:
            print("Warning: ultralytics is not installed. YOLO detection will fail.")
            return None
        
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

def computeOpticalFlowMotion(prevGrayFrame: np.ndarray, grayFrame: np.ndarray, roiState: RoiState) -> float:
    # Crop the current and previous frame to the ROI
    currRoi = grayFrame[roiState.y : roiState.y + roiState.height, roiState.x : roiState.x + roiState.width]
    prevRoi = prevGrayFrame[roiState.y : roiState.y + roiState.height, roiState.x : roiState.x + roiState.width]
    
    # Calculate Dense Optical Flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prevRoi, currRoi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # flow is a 3D array where the 3rd dimension has 2 channels: dx and dy.
    # We are interested in dy (vertical expansion/contraction)
    dy = flow[..., 1]
    
    # We can use the mean of absolute vertical motion
    return float(np.mean(np.abs(dy)))
