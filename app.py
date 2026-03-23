import argparse
import collections
import time
from dataclasses import dataclass

import cv2
import numpy as np

from ultralytics import YOLO

YOLO_MODEL = None


@dataclass
class RoiState:
    x: int
    y: int
    width: int
    height: int
    templateGray: np.ndarray


class BreathEstimator:
    def __init__(
        self,
        breathMinBpm: float,
        breathMaxBpm: float,
        bufferSeconds: float = 18.0,
    ):
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

    def _estimateBandPeak(
        self,
        frequencies: np.ndarray,
        power: np.ndarray,
        minHz: float,
        maxHz: float,
    ) -> tuple[float | None, float]:
        validMask = (frequencies >= minHz) & (frequencies <= maxHz)
        if not np.any(validMask):
            return None, 0.0

        validFrequencies = frequencies[validMask]
        validPower = power[validMask]

        peakIndex = int(np.argmax(validPower))
        peakFrequency = float(validFrequencies[peakIndex])
        peakPower = float(validPower[peakIndex])
        noiseFloor = float(np.median(validPower) + 1e-9)
        confidence = float(np.clip((peakPower / noiseFloor) / 10.0, 0.0, 1.0))
        return peakFrequency * 60.0, confidence

    def estimateBreath(self) -> tuple[float | None, float]:
        if len(self.timeSeries) < 90:
            return None, 0.0

        timeValues = np.array(self.timeSeries, dtype=np.float64)
        motionValues = np.array(self.motionSeries, dtype=np.float64)

        duration = timeValues[-1] - timeValues[0]
        requiredDuration = max(6.0, 2.5 / self.breathMinHz)
        if duration < requiredDuration:
            return None, 0.0

        sampleCount = min(512, int(duration * 30.0))
        if sampleCount < 128:
            return None, 0.0

        uniformTimes = np.linspace(timeValues[0], timeValues[-1], sampleCount)
        uniformSignal = np.interp(uniformTimes, timeValues, motionValues)

        centeredSignal = uniformSignal - np.mean(uniformSignal)
        if np.std(centeredSignal) < 1e-6:
            return None, 0.0

        window = np.hanning(centeredSignal.size)
        windowedSignal = centeredSignal * window

        samplingRate = sampleCount / duration
        spectrum = np.fft.rfft(windowedSignal)
        power = np.abs(spectrum) ** 2
        frequencies = np.fft.rfftfreq(windowedSignal.size, d=1.0 / samplingRate)

        breathBpm, breathConfidence = self._estimateBandPeak(
            frequencies,
            power,
            self.breathMinHz,
            self.breathMaxHz,
        )

        if breathBpm is None:
            self.smoothedBreathBpm = None
        else:
            if self.smoothedBreathBpm is None:
                self.smoothedBreathBpm = breathBpm
            else:
                self.smoothedBreathBpm = 0.2 * breathBpm + 0.8 * self.smoothedBreathBpm

        smoothedBreath = None if self.smoothedBreathBpm is None else float(self.smoothedBreathBpm)
        return smoothedBreath, breathConfidence


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time mouse breath frequency estimation from webcam or video file")
    parser.add_argument("--cameraIndex", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--videoPath", type=str, default="", help="Path to prerecorded video file")
    parser.add_argument("--videoFpsOverride", type=float, default=0.0, help="Manual FPS for video files if metadata is wrong")
    parser.add_argument("--minBpm", type=float, default=60.0, help="Minimum expected breathing rate")
    parser.add_argument("--maxBpm", type=float, default=240.0, help="Maximum expected breathing rate")
    parser.add_argument("--searchMargin", type=int, default=15, help="ROI tracking search margin in pixels")
    parser.add_argument("--bufferSeconds", type=float, default=18.0, help="Signal buffer length in seconds")
    parser.add_argument(
        "--analysisBlurKernel",
        type=int,
        default=1,
        help="Odd Gaussian kernel for analysis blur (1 disables, larger suppresses heartbeat detail)",
    )
    return parser.parse_args()


def selectRoi(frameBgr: np.ndarray) -> RoiState | None:
    box = cv2.selectROI("Select Mouse Chest ROI", frameBgr, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Mouse Chest ROI")

    x, y, width, height = [int(v) for v in box]
    if width < 8 or height < 8:
        return None

    # grayFrame = cv2.cvtColor(frameBgr, cv2.COLOR_BGR2GRAY)
    grayFrame = frameBgr[:, :, 2].copy()
    templateGray = grayFrame[y : y + height, x : x + width].copy()
    return RoiState(x=x, y=y, width=width, height=height, templateGray=templateGray)


def clampRoi(x: int, y: int, width: int, height: int, frameWidth: int, frameHeight: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, frameWidth - width))
    y = max(0, min(y, frameHeight - height))
    return x, y, width, height


def updateRoiByTemplate(grayFrame: np.ndarray, roiState: RoiState, searchMargin: int) -> RoiState:
    frameHeight, frameWidth = grayFrame.shape[:2]

    searchX = max(0, roiState.x - searchMargin)
    searchY = max(0, roiState.y - searchMargin)
    searchX2 = min(frameWidth, roiState.x + roiState.width + searchMargin)
    searchY2 = min(frameHeight, roiState.y + roiState.height + searchMargin)

    searchRegion = grayFrame[searchY:searchY2, searchX:searchX2]
    template = roiState.templateGray

    if searchRegion.shape[0] < template.shape[0] or searchRegion.shape[1] < template.shape[1]:
        return roiState

    result = cv2.matchTemplate(searchRegion, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, maxLocation = cv2.minMaxLoc(result)

    newX = searchX + maxLocation[0]
    newY = searchY + maxLocation[1]
    newX, newY, _, _ = clampRoi(newX, newY, roiState.width, roiState.height, frameWidth, frameHeight)

    newPatch = grayFrame[newY : newY + roiState.height, newX : newX + roiState.width]
    updatedTemplate = cv2.addWeighted(template, 0.92, newPatch, 0.08, 0.0)

    return RoiState(x=newX, y=newY, width=roiState.width, height=roiState.height, templateGray=updatedTemplate)


def computeRoiMotion(grayFrame: np.ndarray, roiState: RoiState) -> float:
    roi = grayFrame[roiState.y : roiState.y + roiState.height, roiState.x : roiState.x + roiState.width]
    roiBlur = cv2.GaussianBlur(roi, (5, 5), 0)
    motionMap = cv2.absdiff(roiBlur, roiState.templateGray)
    return float(np.mean(motionMap))


def drawSignalPanel(frameBgr: np.ndarray, signalValues: collections.deque, panelHeight: int = 110) -> None:
    frameHeight, frameWidth = frameBgr.shape[:2]
    if len(signalValues) < 4:
        return

    values = np.array(signalValues, dtype=np.float32)
    values = values - np.min(values)
    scale = np.max(values) if np.max(values) > 1e-6 else 1.0
    normalized = values / scale

    panelWidth = min(380, frameWidth - 20)
    panelX = 10
    panelY = frameHeight - panelHeight - 10

    cv2.rectangle(frameBgr, (panelX, panelY), (panelX + panelWidth, panelY + panelHeight), (20, 20, 20), -1)
    cv2.rectangle(frameBgr, (panelX, panelY), (panelX + panelWidth, panelY + panelHeight), (180, 180, 180), 1)

    plotCount = min(panelWidth - 8, normalized.size)
    plotValues = normalized[-plotCount:]

    points = []
    for i, value in enumerate(plotValues):
        px = panelX + 4 + i
        py = int(panelY + panelHeight - 6 - value * (panelHeight - 16))
        points.append([px, py])

    pointsArray = np.array(points, dtype=np.int32)
    cv2.polylines(frameBgr, [pointsArray], isClosed=False, color=(90, 220, 120), thickness=1)


def getSampleTime(
    capture: cv2.VideoCapture,
    isVideoFile: bool,
    videoFps: float,
    processedFrameCount: int,
    runStartTime: float,
) -> float:
    if isVideoFile and videoFps > 0.0:
        return processedFrameCount / videoFps

    positionMs = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if positionMs > 0.0:
        return positionMs / 1000.0

    return time.perf_counter() - runStartTime


def getWaitDelayMs(
    isVideoFile: bool,
    videoFps: float,
    processedFrameCount: int,
    playbackStartTime: float,
) -> int:
    if not isVideoFile or videoFps <= 0.0:
        return 1

    expectedElapsed = processedFrameCount / videoFps
    actualElapsed = time.perf_counter() - playbackStartTime
    remainingSeconds = expectedElapsed - actualElapsed

    if remainingSeconds <= 0.0:
        return 1

    return max(1, int(round(remainingSeconds * 1000.0)))


def makeOddKernelSize(kernelSize: int) -> int:
    if kernelSize <= 1:
        return 1
    if kernelSize % 2 == 0:
        return kernelSize + 1
    return kernelSize

def autoDetectRoiYolo(frameBgr: np.ndarray) -> RoiState | None:
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO('yolov8n.pt') 
        
    results = YOLO_MODEL(frameBgr, verbose=False)
    result = results[0]
    
    if len(result.boxes) == 0:
        print("YOLO could not detect any objects in the frame.")
        return None
        
    # Find the bounding box with the highest confidence score
    best_box = None
    best_conf = 0.0
    
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box


    x1, y1, x2, y2 = [int(v) for v in best_box.xyxy[0]] # Extract top-left and bottom-right coordinates
    w = x2 - x1
    h = y2 - y1

    marginX = int(w * 0.1)
    marginY = int(h * 0.1)
    
    newX = x1 + marginX
    newY = y1 + marginY
    newW = max(8, w - (2 * marginX))
    newH = max(8, h - (2 * marginY))

    gray = cv2.cvtColor(frameBgr, cv2.COLOR_BGR2GRAY)
    templateGray = gray[newY : newY + newH, newX : newX + newW].copy()
    
    print(f"YOLO detected target at X:{newX} Y:{newY} W:{newW} H:{newH} (Confidence: {best_conf:.2f})")
    
    return RoiState(x=newX, y=newY, width=newW, height=newH, templateGray=templateGray)

def main() -> None:
    args = parseArgs()
    if args.minBpm <= 0 or args.maxBpm <= 0 or args.maxBpm <= args.minBpm:
        raise ValueError("Use valid BPM bounds: 0 < minBpm < maxBpm.")

    analysisBlurKernel = makeOddKernelSize(args.analysisBlurKernel)

    if args.videoPath:
        capture = cv2.VideoCapture(args.videoPath)
    else:
        capture = cv2.VideoCapture(args.cameraIndex)

    if not capture.isOpened():
        if args.videoPath:
            raise RuntimeError("Could not open video file. Check --videoPath.")
        raise RuntimeError("Could not open camera. Try a different --cameraIndex.")

    isVideoFile = bool(args.videoPath)
    if isVideoFile:
        detectedVideoFps = float(capture.get(cv2.CAP_PROP_FPS))
        videoFps = args.videoFpsOverride if args.videoFpsOverride > 0 else detectedVideoFps
    else:
        videoFps = 0.0

    if isVideoFile and videoFps <= 0.0:
        raise RuntimeError("Could not determine video FPS. Set --videoFpsOverride explicitly.")

    ok, firstFrame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("Could not read first frame from source.")

    # roiState = selectRoi(firstFrame)
    roiState = autoDetectRoiYolo(firstFrame)
    if roiState is None:
        capture.release()
        raise RuntimeError("ROI selection cancelled or too small.")

    # firstGrayFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
    firstGrayFrame = firstFrame[:, :, 2].copy()
    if analysisBlurKernel > 1:
        firstGrayFrame = cv2.GaussianBlur(firstGrayFrame, (analysisBlurKernel, analysisBlurKernel), 0)
    roiState.templateGray = firstGrayFrame[
        roiState.y : roiState.y + roiState.height,
        roiState.x : roiState.x + roiState.width,
    ].copy()

    estimator = BreathEstimator(
        breathMinBpm=args.minBpm,
        breathMaxBpm=args.maxBpm,
        bufferSeconds=args.bufferSeconds,
    )
    runStartTime = time.perf_counter()
    playbackStartTime = time.perf_counter()
    processedFrameCount = 0

    while True:
        ok, frameBgr = capture.read()
        if not ok:
            break

        processedFrameCount += 1
        nowTime = getSampleTime(capture, isVideoFile, videoFps, processedFrameCount, runStartTime)
        # grayFrame = cv2.cvtColor(frameBgr, cv2.COLOR_BGR2GRAY)
        grayFrame = frameBgr[:, :, 2].copy()
        if analysisBlurKernel > 1:
            analysisGrayFrame = cv2.GaussianBlur(grayFrame, (analysisBlurKernel, analysisBlurKernel), 0)
        else:
            analysisGrayFrame = grayFrame

        roiState = updateRoiByTemplate(analysisGrayFrame, roiState, searchMargin=args.searchMargin)
        motionValue = computeRoiMotion(analysisGrayFrame, roiState)
        estimator.addSample(nowTime, motionValue)


        breathBpm, breathConfidence = estimator.estimateBreath()

        cv2.rectangle(
            frameBgr,
            (roiState.x, roiState.y),
            (roiState.x + roiState.width, roiState.y + roiState.height),
            (0, 255, 255),
            2,
        )


        if breathBpm is None:
            breathLabel = "Breath: collecting signal..."
            breathLabelColor = (0, 200, 255)
        else:
            breathLabel = f"Breath: {breathBpm:.1f} BPM | conf {breathConfidence:.2f}"
            breathLabelColor = (60, 240, 120) if breathConfidence >= 0.45 else (0, 180, 255)

        cv2.putText(frameBgr, breathLabel, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, breathLabelColor, 2, cv2.LINE_AA)
        if isVideoFile:
            sourceDetails = f"Source: video @ {videoFps:.2f} fps"
        else:
            sourceDetails = "Source: camera"
        cv2.putText(
            frameBgr,
            f"{sourceDetails} | blur k={analysisBlurKernel} | Keys: q=quit, r=reselect ROI",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        drawSignalPanel(frameBgr, estimator.motionSeries)
        cv2.imshow("Mouse Breath Monitor", frameBgr)

        keyDelayMs = getWaitDelayMs(isVideoFile, videoFps, processedFrameCount, playbackStartTime)
        key = cv2.waitKey(keyDelayMs) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            # newRoiState = selectRoi(frameBgr)
            newRoiState = autoDetectRoiYolo(frameBgr)
            if newRoiState is not None:
                roiState = newRoiState
                roiState.templateGray = analysisGrayFrame[
                    roiState.y : roiState.y + roiState.height,
                    roiState.x : roiState.x + roiState.width,
                ].copy()
                estimator = BreathEstimator(
                    breathMinBpm=args.minBpm,
                    breathMaxBpm=args.maxBpm,
                    bufferSeconds=args.bufferSeconds,
                )
                runStartTime = time.perf_counter()
                playbackStartTime = time.perf_counter()
                processedFrameCount = 0

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
