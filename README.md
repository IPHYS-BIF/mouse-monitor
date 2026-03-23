# Mouse Breath Frequency (Real-Time)

This is a practical MVP app for real-time breath frequency detection from a camera stream of a mouse.

## What it does
- Captures live video from a webcam.
- Lets you select a chest/abdomen ROI on the mouse.
- Tracks that ROI frame-by-frame using template matching.
- Extracts a motion signal and estimates dominant breathing frequency using FFT.
- Shows live `BPM` and a confidence score.

## Setup
The script requires several dependencies listed in `requirements.txt`. It is advisable to install them in a fresh conda environment. 

```powershell
cd "d:\OneDrive - IEM\000_inbox\260312_breath_measurement"
conda create -n mouseBreathMonitor python=3.13 -y
pip install -r requirements.txt
```


## Run
```powershell
python app.py --cameraIndex 0 --minBpm 60 --maxBpm 360
```

Use a prerecorded video file:
```powershell
python app.py --videoPath "data\mouse_breath.mp4" --minBpm 60 --maxBpm 360
```

If the video has wrong/missing FPS metadata, force FPS explicitly:
```powershell
python app.py --videoPath "data\mouse_breath.mp4" --videoFpsOverride 30 --minBpm 10 --maxBpm 360
```

To suppress fine motion and focus on breathing, increase analysis blur:
```powershell
python app.py --videoPath "data\mouse_breath.mp4" --videoFpsOverride 30 --minBpm 10 --maxBpm 240 --analysisBlurKernel 15
```

Notes:
- `--analysisBlurKernel 1` disables analysis blur.
- Use odd values like `9`, `11`, `15`, `21`; larger values suppress more fine detail.


## Real-time breath frequency estimation
This app estimates the breathing frequency from the ROI signal.

Example for 30 fps mouse video:
```powershell
python app.py --videoPath "data\mouse_breath.mp4" --videoFpsOverride 30 --minBpm 10 --maxBpm 180 --analysisBlurKernel 11
```

Notes:
- Use `--minBpm` and `--maxBpm` to set the expected breathing rate range.
- Increase `--analysisBlurKernel` to suppress fine motion and focus on breathing.

## Controls
- `q`: quit
- `r`: reselect ROI

## Tips for better signal quality
- Use stable camera mounting.
- Keep the mouse body large in frame.
- Select ROI over visible breathing motion (thorax or upper abdomen).
- Use good, steady lighting.
- Reduce cage/bed motion in the ROI.

## Current limitations
- This is non-contact motion-based estimation, not a clinical-grade respiratory monitor.
- Large full-body motion can temporarily degrade estimation.
- FFT peak picking is simple and can drift in challenging scenes.


## Synthetic real-time stream generator (for camera-based validation)
You can generate a known-motion video stream on your monitor, then point your webcam at the screen and run `app.py` in parallel.

Start generator (known breathing + heartbeat frequencies):
```powershell
python signal_video_generator.py --fps 30 --breathBpm 60 --heartBpm 420 --breathAmplitude 24 --heartAmplitude 3 --fullscreen
```

Add irregularities and extra movement:
```powershell
python signal_video_generator.py --fps 30 --breathBpm 70 --heartBpm 480 --breathFreqJitterBpm 8 --heartFreqJitterBpm 25 --irregularityRateHz 0.2 --irregularityAmplitude 24 --globalShakeAmplitude 5 --noiseStd 5 --fullscreen
```


Then run monitor against webcam feed (separate terminal):
```powershell
python app.py --cameraIndex 0 --minBpm 10 --maxBpm 180
```

Generator keys:
- `q`: quit
- `f`: toggle fullscreen
- `g`: toggle calibration grid
