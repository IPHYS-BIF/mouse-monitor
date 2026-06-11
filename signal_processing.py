import collections
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

class AdvancedBreathEstimator:
    def __init__(self, breathMinBpm: float = 60.0, breathMaxBpm: float = 240.0, bufferSeconds: float = 18.0):
        self.breathMinHz = breathMinBpm / 60.0
        self.breathMaxHz = breathMaxBpm / 60.0
        self.bufferSeconds = bufferSeconds
        self.timeSeries = collections.deque()
        self.motionSeries = collections.deque()
        self.smoothedBreathBpm = None

    def update_limits(self, min_bpm: float, max_bpm: float) -> None:
        # Add bias to the limits to prevent clipping valid signals that are just outside the alarm range
        bias_bpm = 30.0
        self.breathMinHz = max(5.0, min_bpm - bias_bpm) / 60.0
        self.breathMaxHz = (max_bpm + bias_bpm) / 60.0

    def addSample(self, sampleTime: float, motionValue: float) -> None:
        self.timeSeries.append(sampleTime)
        self.motionSeries.append(motionValue)
        while self.timeSeries and (sampleTime - self.timeSeries[0]) > self.bufferSeconds:
            self.timeSeries.popleft()
            self.motionSeries.popleft()

    def estimateBreath(self) -> tuple[float | None, float]:
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
        
        peaks, _ = find_peaks(filteredSignal, distance=min_distance_samples)
        
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
