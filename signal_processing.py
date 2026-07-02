import collections
import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt, find_peaks, detrend

class AdvancedBreathEstimator:
    def __init__(self, method = "default", breathMinBpm: float = 60.0, breathMaxBpm: float = 240.0, bufferSeconds: float = 30.0):
        self.breathMinHz = breathMinBpm / 60.0
        self.breathMaxHz = breathMaxBpm / 60.0
        self.bufferSeconds = bufferSeconds
        self.timeSeries = collections.deque()
        self.motionSeries = collections.deque()
        self.method = method
        self.smoothedBreathBpm = None
        self.last_update_time = None  # Track when BPM was last updated
        self.update_interval = 15.0  # Update BPM every 15 seconds (75% window overlap)
        self.last_estimate_success = False  # Track if last estimate was valid (for adaptive retry)
        # correlation method buffers
        self.frameTimeSeries = collections.deque()
        self.frameSeries = collections.deque()
        self.previousThumb = None  # For temporal pixel filtering
        self.pixelMask = None  # Mask of pixels that change over time
        self.last_peaks = []  # Store last detected peak times
        self.last_reference_time = None  # Current time for reference
        # Rolling pixel-change history for static detection
        self.pixelChangeMagnitude = collections.deque()
        self.pixelChangeTime = collections.deque()

    def addSample(self, sampleTime: float, motionValue: float) -> None:
        self.timeSeries.append(sampleTime)
        self.motionSeries.append(motionValue)
        while self.timeSeries and (sampleTime - self.timeSeries[0]) > self.bufferSeconds:
            self.timeSeries.popleft()
            self.motionSeries.popleft()

    def addFrame(self, sampleTime: float, roiGray: np.ndarray) -> None:
        """For the 'correlation' method: store a downsampled ROI frame alongside its timestamp.
        
        Also builds a mask of pixels that change over time, filtering out static background.
        """
        h, w = roiGray.shape[:2]
        # Downsample to at most 16x16 with simple strided slicing (no extra dependency)
        step_h = max(1, h // 16)
        step_w = max(1, w // 16)
        thumb = roiGray[::step_h, ::step_w][:16, :16].astype(np.float32).flatten()
        
        # Update temporal pixel mask: track which pixels change over time
        if self.previousThumb is not None:
            temporal_diff = np.abs(thumb - self.previousThumb)
            mean_change = float(np.mean(temporal_diff))
            self.pixelChangeMagnitude.append(mean_change)
            self.pixelChangeTime.append(sampleTime)
            # Trim change history to buffer window
            while self.pixelChangeTime and (sampleTime - self.pixelChangeTime[0]) > self.bufferSeconds:
                self.pixelChangeTime.popleft()
                self.pixelChangeMagnitude.popleft()
            if self.pixelMask is None:
                self.pixelMask = temporal_diff > 1.0
            else:
                # Accumulate: mark pixels as changing if they differ in any frame
                self.pixelMask |= (temporal_diff > 1.0)
        
        self.previousThumb = thumb.copy()
        self.frameSeries.append(thumb)
        self.frameTimeSeries.append(sampleTime)
        while self.frameTimeSeries and (sampleTime - self.frameTimeSeries[0]) > self.bufferSeconds:
            self.frameTimeSeries.popleft()
            self.frameSeries.popleft()

    def has_enough_data(self) -> bool:
        """Returns True when enough time has elapsed for the algorithm to produce a result."""
        if len(self.timeSeries) < 90:
            return False
        duration = self.timeSeries[-1] - self.timeSeries[0]
        return duration >= max(6.0, 2.5 / self.breathMinHz)

    def _is_static(self, window: float = 10.0, threshold: float = 0.05) -> bool:
        """Returns True if pixel intensity changes over the last `window` seconds are negligible.
        
        Only return True for genuinely static scenes (ROI in background).
        Use low threshold (0.05) to avoid false positives with subtle breathing.
        """
        if len(self.pixelChangeTime) < 30:
            return False  # Not enough data yet
        current_time = self.pixelChangeTime[-1]
        cutoff = current_time - window
        recent = [m for t, m in zip(self.pixelChangeTime, self.pixelChangeMagnitude) if t >= cutoff]
        if len(recent) < 20:
            return False
        return float(np.mean(recent)) < threshold

    def estimateBreath(self) -> tuple[float | None, float]:
        if self._is_static():
            self.last_peaks = []
            self.last_estimate_success = False
            return None, 0.0
        
        # Adaptive retry: 5s if signal lost, 15s if signal found
        # Fast recovery when breathing resumes, stable interval when stable
        current_time = self.timeSeries[-1] if self.timeSeries else 0
        throttle_interval = 5.0 if not self.last_estimate_success else self.update_interval
        
        if self.last_update_time is not None and (current_time - self.last_update_time) < throttle_interval:
            # Not enough time has passed — return last known BPM
            return self.smoothedBreathBpm, 0.0
        
        # Time to update — run estimation and record update time
        self.last_update_time = current_time
        
        if self.method == "correlation":
            result = self._estimateBreathCorrelation()
        else:
            result = self._estimateBreathDefault()
        
        # Track success for adaptive retry logic
        self.last_estimate_success = (result[0] is not None)
        return result

    def _estimateBreathCorrelation(self) -> tuple[float | None, float]:
        """FFT-based autocorrelation on ROI frames.

        Each ROI frame is projected onto the mean-frame template to produce a
        1D scalar signal over time.  The power spectral density (|FFT|²) of
        that signal is the autocorrelation in the frequency domain; the
        dominant peak inside the allowed breathing band is the breath rate.
        """
        if len(self.frameSeries) < 90:
            return None, 0.0

        times = np.array(self.frameTimeSeries, dtype=np.float64)
        frames = np.array(self.frameSeries, dtype=np.float64)  # (N, pixels)

        duration = times[-1] - times[0]
        if duration < max(6.0, 2.5 / self.breathMinHz):
            return None, 0.0

        # Apply temporal pixel mask: only use pixels that change over time
        if self.pixelMask is not None and np.any(self.pixelMask):
            # Filter frames to only include changing pixels
            frames_filtered = frames[:, self.pixelMask]
        else:
            frames_filtered = frames
        
        # Variance-weighted pixel projection:
        # Weight each pixel by its temporal variance so that pixels that actually
        # move (breathing pixels) dominate the signal, while static background
        # pixels contribute almost nothing. This is far more selective than
        # projecting onto the mean frame (which weights by brightness, not motion).
        pixel_var = np.var(frames_filtered, axis=0)
        total_var = np.sum(pixel_var)
        if total_var < 1e-6:
            return None, 0.0
        weights = pixel_var / total_var          # normalised, sum = 1
        signal = frames_filtered @ weights       # shape: (N,)
        signal = signal - np.mean(signal)

        if np.std(signal) < 1e-6:
            return None, 0.0

        # Resample to a uniform time grid — use up to 1024 points for better frequency resolution
        sampleCount = min(1024, int(duration * 30.0))
        uniformTimes = np.linspace(times[0], times[-1], sampleCount)
        uniformSignal = np.interp(uniformTimes, times, signal)
        samplingRate = sampleCount / duration

        # Normalize signal
        signal_norm = uniformSignal - np.mean(uniformSignal)
        signal_std = np.std(signal_norm)
        if signal_std < 1e-6:
            return None, 0.0
        signal_norm = signal_norm / signal_std

        # Remove baseline trends (slow lighting changes, drift) using detrend
        # This removes linear trends up to the timescale of ~1-2 seconds
        signal_detrended = detrend(signal_norm, type='linear')

        # Apply bandpass filter to clean signal before FFT
        # 0.5 Hz = 30 BPM, 5 Hz = 300 BPM — breathing band
        nyq = samplingRate / 2.0
        low = 0.5 / nyq
        high = 5.0 / nyq
        if low > 0 and high < 1 and low < high:
            try:
                sos = butter(4, [low, high], btype='band', output='sos')
                signal_filtered = sosfiltfilt(sos, signal_detrended)
            except (ValueError, RuntimeError):
                signal_filtered = signal_detrended
        else:
            signal_filtered = signal_detrended

        # Compute FFT power spectrum — use PSD to find dominant frequency directly.
        # This is more robust than argmax on autocorrelation lags, which is biased
        # toward short lags (high frequencies) in low-SNR signals.
        spectrum = np.fft.rfft(signal_filtered)
        psd = np.abs(spectrum) ** 2
        freqs = np.fft.rfftfreq(sampleCount, d=1.0 / samplingRate)

        # Restrict to breathing band
        band_mask = (freqs >= self.breathMinHz) & (freqs <= self.breathMaxHz)
        if not np.any(band_mask):
            return None, 0.0

        psd_band = psd[band_mask]
        freqs_band = freqs[band_mask]

        total_band_power = np.sum(psd_band)
        if total_band_power < 1e-12:
            return None, 0.0

        # Find the LOWEST-frequency significant peak in the band.
        # The fundamental is always the lowest harmonic; argmax biases toward
        # higher-frequency harmonics in low-SNR signals, so we scan low→high.
        max_power_in_band = np.max(psd_band)
        significance_threshold = 0.20 * max_power_in_band  # local peak must be ≥20% of band max
        peak_idx = None
        for i in range(1, len(psd_band) - 1):
            if (psd_band[i] >= significance_threshold and
                    psd_band[i] >= psd_band[i - 1] and
                    psd_band[i] >= psd_band[i + 1]):
                peak_idx = i
                break
        if peak_idx is None:
            peak_idx = int(np.argmax(psd_band))  # fallback

        peak_power = psd_band[peak_idx]
        snr_confidence = float(peak_power / total_band_power)

        # Require reasonable SNR: peak must hold ≥2% of total band power
        # Low threshold to catch even weak breathing signals
        # Preprocessing (bandpass + detrend) already filters noise
        if snr_confidence < 0.02:
            return None, 0.0

        peak_freq = float(freqs_band[peak_idx])

        if peak_freq <= 0:
            return None, 0.0

        bpm = peak_freq * 60.0
        period = 1.0 / peak_freq

        # Red line visualization removed
        self.last_peaks = []

        # Stable BPM tracking with 1-minute sliding window
        # Updates every 30 seconds (50% overlap), providing natural temporal smoothing
        if self.smoothedBreathBpm is None:
            self.smoothedBreathBpm = bpm
        else:
            # Light exponential smoothing between window updates
            self.smoothedBreathBpm = 0.3 * bpm + 0.7 * self.smoothedBreathBpm

        return float(self.smoothedBreathBpm), snr_confidence

    def _estimateBreathDefault(self) -> tuple[float | None, float]:
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
        
        # Center signal
        centeredSignal = uniformSignal - np.mean(uniformSignal)
        if np.std(centeredSignal) < 1e-6:
            return None, 0.0

        samplingRate = sampleCount / duration
        
        # Butterworth Bandpass Filter
        nyq = 0.5 * samplingRate
        low = self.breathMinHz / nyq
        high = self.breathMaxHz / nyq
        
        # Ensure valid filter parameters
        if low <= 0 or high >= 1 or low >= high:
            return None, 0.0
            
        b, a = butter(2, [low, high], btype='band')
        try:
            filteredSignal = filtfilt(b, a, centeredSignal)
        except ValueError:
            return None, 0.0
            
        # Peak Detection
        # Minimum distance between peaks depends on the max allowed frequency
        min_distance_samples = int(samplingRate / self.breathMaxHz)
        
        # Calculate height threshold to filter out noise/ripples
        # Only keep peaks that are in the upper 30% of the signal range
        signal_min = np.min(filteredSignal)
        signal_max = np.max(filteredSignal)
        signal_range = signal_max - signal_min
        height_threshold = signal_min + 0.3 * signal_range  # Top 30% of amplitude
        
        peaks, _ = find_peaks(filteredSignal, distance=min_distance_samples, height=height_threshold)
        
        # Store detected peak times for visualization in time domain
        if len(peaks) > 0:
            # Get peak times from uniformTimes
            self.last_peaks = [float(uniformTimes[i]) for i in peaks]
            self.last_reference_time = float(timeValues[-1])  # Current time reference
        else:
            self.last_peaks = []
            self.last_reference_time = float(timeValues[-1])
        
        if len(peaks) < 2:
            return None, 0.0
            
        peak_times = uniformTimes[peaks]
        intervals = np.diff(peak_times)
        
        if len(intervals) == 0:
            return None, 0.0
            
        mean_interval = np.mean(intervals)
        if mean_interval == 0:
            return None, 0.0
            
        breathBpm = (1.0 / mean_interval) * 60.0
        
        # Filter again by limits to be safe
        if breathBpm < (self.breathMinHz * 60.0) or breathBpm > (self.breathMaxHz * 60.0):
            return None, 0.0

        confidence = 1.0 # Pseudo confidence

        if self.smoothedBreathBpm is None:
            self.smoothedBreathBpm = breathBpm
        else:
            self.smoothedBreathBpm = 0.2 * breathBpm + 0.8 * self.smoothedBreathBpm

        return float(self.smoothedBreathBpm), confidence
    
    def get_last_peaks(self):
        """Return the last detected peak times, reference time, and method type.
        Returns: (peak_times_list, reference_time, method)
        """
        return (self.last_peaks, self.last_reference_time, self.method)
