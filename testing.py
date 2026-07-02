"""
breath_rate.py
--------------
Estimates breathing frequency from a video of a constrained mouse.

Usage:
    python breath_rate.py <video_path> [--roi x y w h] [--fps FPS]

Arguments:
    video_path       Path to the input video file.
    --roi x y w h    Optional: manually specify the Region of Interest as
                     x, y (top-left corner), width, height (in pixels).
                     If omitted, you will be prompted to draw the ROI
                     interactively on the first frame.
    --fps FPS        Override the video frame rate (useful for cameras that
                     report an incorrect FPS).

How it works:
    1. The user selects (or provides) a ROI covering the thorax/flank of the mouse.
    2. For each frame, the mean pixel intensity of the ROI is recorded → a 1-D
       time series that captures the chest expansion/contraction cycle.
    3. The signal is detrended and bandpass-filtered (Butterworth, 1–10 Hz,
       i.e. 60–600 breaths/min — covering the full rodent range).
    4. A short sliding window (default 5 s) is analysed with Welch's method to
       get a robust, up-to-date frequency estimate.
    5. The live video is displayed with the ROI rectangle and the current
       breathing rate (breaths/min) overlaid.

Requirements:
    pip install opencv-python numpy scipy matplotlib
"""

import argparse
import sys
import collections
import time

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.signal import detrend as sp_detrend


# ── Physiological constants for rodents (adjust if needed) ──────────────────
RR_MIN_HZ   = 1.0    # 60  breaths/min  – lower bound for bandpass
RR_MAX_HZ   = 10.0   # 600 breaths/min  – upper bound for bandpass
WINDOW_SEC  = 5.0    # sliding analysis window length (seconds)
STEP_SEC    = 0.5    # how often the RR estimate is refreshed (seconds)


# ── Signal-processing helpers ────────────────────────────────────────────────

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)   # stay below Nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def estimate_rr_welch(signal, fs):
    """Return the dominant frequency (Hz) in the respiration band via Welch PSD.

    Uses the full signal length as nperseg (finest native resolution) and
    zero-pads by 4x (nfft) to interpolate the spectrum, avoiding coarse grid
    artefacts that cause estimates to snap to rounded values.

    Frequency resolution = fs / nperseg  (native)
    Interpolated step    = fs / nfft     (display / peak-finding)
    """
    if len(signal) < 2:
        return 0.0
    nperseg = len(signal)           # use the whole window → Δf = fs/N
    nfft    = nperseg * 4           # 4× zero-padding for sub-bin interpolation
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, nfft=nfft)
    # Restrict to the physiological band
    mask = (freqs >= RR_MIN_HZ) & (freqs <= RR_MAX_HZ)
    if not mask.any():
        return 0.0
    dominant_freq = freqs[mask][np.argmax(psd[mask])]
    return dominant_freq


# ── ROI selection ────────────────────────────────────────────────────────────

def select_roi_interactive(frame):
    """Let the user drag a rectangle on the first frame. Returns (x, y, w, h)."""
    print("\n[INFO] Draw the ROI (thorax/flank region) on the window that opens.")
    print("       Click-and-drag, then press ENTER or SPACE to confirm, ESC to cancel.\n")
    roi = cv2.selectROI("Select ROI – press ENTER to confirm", frame,
                        fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI – press ENTER to confirm")
    if roi == (0, 0, 0, 0):
        print("[ERROR] No ROI selected. Exiting.")
        sys.exit(1)
    return roi   # (x, y, w, h)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_overlay(frame, roi, rr_bpm, signal_buf, fs):
    """Draw ROI rectangle, BPM text, and a mini waveform on the frame."""
    x, y, w, h = roi
    height, width = frame.shape[:2]

    # ROI rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, "ROI", (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    # Breathing rate text
    if rr_bpm > 0:
        bpm_text = f"Breath rate: {rr_bpm:.1f} bpm"
        color = (0, 200, 255)
    else:
        bpm_text = "Breath rate: estimating..."
        color = (100, 100, 100)

    cv2.putText(frame, bpm_text, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # Mini waveform in the bottom strip
    if len(signal_buf) > 10:
        sig = np.array(signal_buf)
        sig = sig - sig.mean()
        mx = np.abs(sig).max() or 1.0
        sig /= mx

        strip_h = 60
        strip_y = height - strip_h - 5
        strip_x = 10
        strip_w = min(width - 20, 400)

        cv2.rectangle(frame, (strip_x, strip_y),
                      (strip_x + strip_w, strip_y + strip_h),
                      (30, 30, 30), -1)

        pts = np.linspace(0, len(sig) - 1, strip_w).astype(int)
        sub = sig[pts]
        xs = np.arange(strip_x, strip_x + strip_w)
        ys = (strip_y + strip_h // 2 - sub * (strip_h // 2 - 4)).astype(int)
        pts2d = np.column_stack([xs, ys])
        cv2.polylines(frame, [pts2d], False, (0, 200, 255), 1, cv2.LINE_AA)

    return frame


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Estimate breathing rate from a constrained-mouse video.")
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="ROI: top-left x y, width w, height h.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override the frame rate reported by the camera.")
    args = parser.parse_args()

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("[WARNING] FPS not detected; defaulting to 30. Use --fps to override.")
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {args.video}")
    print(f"[INFO] FPS: {fps:.2f}   Total frames: {total_frames}")

    # ── Read first frame & ROI ────────────────────────────────────────────────
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame.")
        sys.exit(1)

    if args.roi:
        roi = tuple(args.roi)
        print(f"[INFO] Using provided ROI: {roi}")
    else:
        roi = select_roi_interactive(first_frame.copy())
        print(f"[INFO] ROI selected: {roi}")

    rx, ry, rw, rh = roi

    # ── Bandpass filter coefficients ──────────────────────────────────────────
    b, a = butter_bandpass(RR_MIN_HZ, RR_MAX_HZ, fps)

    # ── Buffers ───────────────────────────────────────────────────────────────
    window_frames = int(WINDOW_SEC * fps)
    step_frames   = max(1, int(STEP_SEC * fps))

    raw_buf     = collections.deque(maxlen=window_frames)  # raw intensity
    display_buf = collections.deque(maxlen=window_frames)  # filtered signal (for waveform)
    rr_bpm      = 0.0
    frame_count = 0

    # Rewind and include first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_period_ms = 1000.0 / fps   # target milliseconds per frame

    print("\n[INFO] Processing — press Q in the video window to quit.\n")

    while True:
        frame_start = time.perf_counter()   # start wall-clock for this frame

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ── Extract ROI mean intensity ────────────────────────────────────────
        roi_patch = frame[ry: ry + rh, rx: rx + rw]
        mean_intensity = roi_patch.mean()      # scalar per frame
        raw_buf.append(mean_intensity)

        # ── Estimate RR every STEP_SEC seconds (once we have enough data) ─────
        if len(raw_buf) >= window_frames and (frame_count % step_frames == 0):
            sig = np.array(raw_buf, dtype=float)
            sig = sp_detrend(sig)              # remove linear drift

            # Bandpass filter (zero-phase)
            if len(sig) > 3 * (len(b) - 1):   # filtfilt needs length > 3*order
                sig_filt = filtfilt(b, a, sig)
            else:
                sig_filt = sig

            dominant_hz = estimate_rr_welch(sig_filt, fps)
            rr_bpm = dominant_hz * 60.0

            # Update the display buffer with the filtered signal
            display_buf.clear()
            display_buf.extend(sig_filt.tolist())

        # ── Draw & show ───────────────────────────────────────────────────────
        vis = frame.copy()
        vis = draw_overlay(vis, roi, rr_bpm, display_buf, fps)

        # Show frame number / total in corner
        info = f"Frame {frame_count}/{total_frames}"
        cv2.putText(vis, info, (12, vis.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("Rodent Breathing Monitor", vis)

        # ── Throttle playback to the true frame rate ──────────────────────────
        # Compute how long this frame took to process, then wait only the
        # remaining time so that each frame takes exactly 1/fps seconds total.
        elapsed_ms = (time.perf_counter() - frame_start) * 1000
        wait_ms = max(1, int(frame_period_ms - elapsed_ms))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q") or key == 27:   # Q or ESC
            print("[INFO] User quit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Done. Last estimated breath rate: {rr_bpm:.1f} bpm")


if __name__ == "__main__":
    main()