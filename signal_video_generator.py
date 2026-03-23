import argparse
import math
import time

import cv2
import numpy as np


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time synthetic motion stream for breath/heart monitor testing")
    parser.add_argument("--width", type=int, default=1280, help="Output frame width")
    parser.add_argument("--height", type=int, default=720, help="Output frame height")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback frame rate")

    parser.add_argument("--breathBpm", type=float, default=80.0, help="Base breathing rate")
    parser.add_argument("--heartBpm", type=float, default=420.0, help="Base heartbeat rate")
    parser.add_argument("--breathAmplitude", type=float, default=22.0, help="Breathing displacement amplitude in pixels")
    parser.add_argument("--heartAmplitude", type=float, default=2.5, help="Heartbeat displacement amplitude in pixels")

    parser.add_argument("--breathFreqJitterBpm", type=float, default=0.0, help="Breathing BPM random jitter (+/-)")
    parser.add_argument("--heartFreqJitterBpm", type=float, default=0.0, help="Heartbeat BPM random jitter (+/-)")
    parser.add_argument("--phaseNoise", type=float, default=0.015, help="Random phase noise strength")

    parser.add_argument("--irregularityRateHz", type=float, default=0.08, help="Expected irregular movement event rate")
    parser.add_argument("--irregularityAmplitude", type=float, default=18.0, help="Irregular movement event amplitude")
    parser.add_argument("--irregularityDecay", type=float, default=2.5, help="Exponential decay rate for irregular events")

    parser.add_argument("--globalShakeAmplitude", type=float, default=3.0, help="Global motion amplitude in pixels")
    parser.add_argument("--globalShakeHz", type=float, default=1.4, help="Global motion frequency")
    parser.add_argument("--noiseStd", type=float, default=3.0, help="Pixel noise standard deviation")

    parser.add_argument("--grid", action="store_true", help="Draw subtle calibration grid")
    parser.add_argument("--seed", type=int, default=20260312, help="Random seed for reproducible sequence")
    parser.add_argument("--windowName", type=str, default="Synthetic Motion Stream", help="Display window title")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")

    return parser.parse_args()


def drawBackground(frame: np.ndarray, grid: bool) -> None:
    height, width = frame.shape[:2]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    gradient = (35 + 25 * y).astype(np.uint8)
    frame[:, :, 0] = gradient[:, None]
    frame[:, :, 1] = (gradient[:, None] + 10).clip(0, 255)
    frame[:, :, 2] = (gradient[:, None] + 15).clip(0, 255)

    if grid:
        gridColor = (55, 60, 65)
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), gridColor, 1, cv2.LINE_AA)
        for yCoord in range(0, height, 80):
            cv2.line(frame, (0, yCoord), (width, yCoord), gridColor, 1, cv2.LINE_AA)


def drawTestTarget(frame: np.ndarray, centerX: int, centerY: int, radius: int, color: tuple[int, int, int]) -> None:
    cv2.circle(frame, (centerX, centerY), radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (centerX, centerY), radius + 2, (220, 220, 220), 2, cv2.LINE_AA)


def drawOverlay(
    frame: np.ndarray,
    breathBpm: float,
    heartBpm: float,
    jitterBreath: float,
    jitterHeart: float,
    irregularityValue: float,
    fps: float,
) -> None:
    line1 = f"Breath target: {breathBpm:.1f} BPM | Heart target: {heartBpm:.1f} BPM"
    line2 = (
        f"Jitter breath: {jitterBreath:+.2f} BPM | "
        f"Jitter heart: {jitterHeart:+.2f} BPM | "
        f"Irregularity: {irregularityValue:+.2f} px"
    )
    line3 = f"Playback: {fps:.2f} fps | Keys: q=quit, f=fullscreen, g=grid"

    cv2.rectangle(frame, (12, 12), (frame.shape[1] - 12, 96), (10, 10, 10), -1)
    cv2.rectangle(frame, (12, 12), (frame.shape[1] - 12, 96), (160, 160, 160), 1)
    cv2.putText(frame, line1, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (90, 220, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 200, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, line3, (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1, cv2.LINE_AA)


def main() -> None:
    args = parseArgs()
    if args.fps <= 0:
        raise ValueError("fps must be > 0")

    rng = np.random.default_rng(args.seed)
    windowName = args.windowName
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    if args.fullscreen:
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(windowName, args.width, args.height)

    showGrid = bool(args.grid)

    phaseBreath = 0.0
    phaseHeart = 0.0
    irregularityValue = 0.0

    previousTime = time.perf_counter()
    streamStartTime = previousTime
    frameIndex = 0

    while True:
        currentTime = time.perf_counter()
        deltaTime = max(1e-4, currentTime - previousTime)
        previousTime = currentTime

        jitterBreath = float(rng.uniform(-args.breathFreqJitterBpm, args.breathFreqJitterBpm))
        jitterHeart = float(rng.uniform(-args.heartFreqJitterBpm, args.heartFreqJitterBpm))

        breathHz = max(0.01, (args.breathBpm + jitterBreath) / 60.0)
        heartHz = max(0.01, (args.heartBpm + jitterHeart) / 60.0)

        phaseBreath += 2.0 * math.pi * breathHz * deltaTime + float(rng.normal(0.0, args.phaseNoise))
        phaseHeart += 2.0 * math.pi * heartHz * deltaTime + float(rng.normal(0.0, args.phaseNoise * 0.7))

        irregularityValue *= math.exp(-args.irregularityDecay * deltaTime)
        if rng.random() < args.irregularityRateHz * deltaTime:
            irregularityValue += float(rng.normal(0.0, args.irregularityAmplitude))

        elapsed = currentTime - streamStartTime
        globalShakeX = args.globalShakeAmplitude * math.sin(2.0 * math.pi * args.globalShakeHz * elapsed)
        globalShakeY = args.globalShakeAmplitude * math.sin(2.0 * math.pi * args.globalShakeHz * elapsed + 0.9)

        targetDx = (
            args.breathAmplitude * math.sin(phaseBreath)
            + args.heartAmplitude * math.sin(phaseHeart)
            + irregularityValue
            + globalShakeX
        )
        targetDy = (
            0.35 * args.breathAmplitude * math.sin(phaseBreath + 0.6)
            + 0.25 * args.heartAmplitude * math.sin(phaseHeart + 1.3)
            + 0.5 * irregularityValue
            + globalShakeY
        )

        frame = np.empty((args.height, args.width, 3), dtype=np.uint8)
        drawBackground(frame, showGrid)

        centerX = int(args.width * 0.5 + targetDx)
        centerY = int(args.height * 0.58 + targetDy)
        centerX = int(np.clip(centerX, 80, args.width - 80))
        centerY = int(np.clip(centerY, 120, args.height - 80))

        # Outer low-frequency body movement target.
        drawTestTarget(frame, centerX, centerY, 54, (110, 190, 250))
        # Inner high-frequency component marker.
        drawTestTarget(
            frame,
            centerX + int(8 * math.sin(phaseHeart)),
            centerY + int(6 * math.cos(phaseHeart)),
            14,
            (80, 80, 240),
        )

        drawOverlay(
            frame,
            args.breathBpm,
            args.heartBpm,
            jitterBreath,
            jitterHeart,
            irregularityValue,
            args.fps,
        )

        if args.noiseStd > 0:
            noise = rng.normal(0.0, args.noiseStd, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        cv2.imshow(windowName, frame)

        frameIndex += 1
        expectedElapsed = frameIndex / args.fps
        actualElapsed = time.perf_counter() - streamStartTime
        waitMs = max(1, int(round((expectedElapsed - actualElapsed) * 1000.0)))

        key = cv2.waitKey(waitMs) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            args.fullscreen = not args.fullscreen
            mode = cv2.WINDOW_FULLSCREEN if args.fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, mode)
            if not args.fullscreen:
                cv2.resizeWindow(windowName, args.width, args.height)
        if key == ord("g"):
            showGrid = not showGrid

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
