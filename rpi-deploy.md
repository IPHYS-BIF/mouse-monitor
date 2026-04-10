You are 100% correct to be cautious. Pulling the power cord on a Raspberry Pi while it's running is the fastest way to corrupt the SD card and lose all your data (and your script!). 

Because your Pi 5 will be running as a "headless" kiosk appliance (meaning no desktop, no start menu, just your app), we need to handle the boot and shutdown processes gracefully.

Here is the best-practice approach for a modern Raspberry Pi OS (Bookworm) appliance.

### Part 1: The Safe Shutdown (UI Integration)
Since you have a touch screen, the most elegant solution is to add a "Power Off" button directly to your PyQt6 dashboard. 

Because fat-fingering a touch screen and accidentally shutting down the system mid-experiment would be a disaster, we should wrap the shutdown command in a confirmation popup.

**1. Add this import to the top of your `app.py`:**
```python
import os
from PyQt6.QtWidgets import QMessageBox
```

**2. Add the button to your `init_ui` function:**
Find where we added the `btn_record` in the Top Bar (around line 225), and add a shutdown button right next to it:
```python
        # Existing Record Button
        self.btn_record = QPushButton("RECORD SESSION")
        self.btn_record.setObjectName("RecordButton")
        self.btn_record.setFixedSize(160, 45)
        
        # NEW Shutdown Button
        self.btn_shutdown = QPushButton("POWER OFF")
        self.btn_shutdown.setFixedSize(120, 45)
        self.btn_shutdown.setStyleSheet("""
            background-color: #2a3439; 
            color: white; 
            font-weight: bold; 
            border-radius: 6px;
        """)
        self.btn_shutdown.clicked.connect(self.confirm_shutdown)

        # Add them both to the layout
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_record)
        top_bar.addWidget(self.btn_shutdown) # <-- Added here
```

**3. Add the shutdown logic:**
Add this function anywhere inside your `MouseTrackerDashboard` class:
```python
    def confirm_shutdown(self):
        # Create a large, touch-friendly confirmation dialog
        msg = QMessageBox(self)
        msg.setWindowTitle("System Shutdown")
        msg.setText("Are you sure you want to power off the Station?")
        msg.setStyleSheet("QLabel { font-size: 18px; } QPushButton { font-size: 16px; padding: 10px; min-width: 100px; }")
        
        yes_btn = msg.addButton("Yes, Power Off", QMessageBox.ButtonRole.YesRole)
        no_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.NoRole)
        
        msg.exec()
        
        if msg.clickedButton() == yes_btn:
            self.lbl_status.setText("SHUTTING DOWN...")
            # Close the app safely to free the camera
            self.close() 
            # Tell the Raspberry Pi Linux kernel to halt
            os.system("sudo shutdown -h now") 
```

*(Note: The Raspberry Pi 5 actually has a physical power button built directly onto the circuit board. If you press it once while the Pi is on, it initiates a safe, soft shutdown automatically! But having it in the UI is usually better if the Pi is inside an enclosure).*

---

### Part 2: Autostart on Boot
The Raspberry Pi 5 uses a modern display manager called Wayland. The absolute most reliable way to make a GUI app run at startup on Wayland is using the standard Linux `.desktop` file method.

Here is how to set it up:

**1. Open a terminal on your Raspberry Pi.**
First, make sure the hidden `autostart` folder exists:
```bash
mkdir -p ~/.config/autostart
```

**2. Create a new `.desktop` file:**
We will use the `nano` text editor to create a shortcut file that the system reads when it boots up.
```bash
nano ~/.config/autostart/mousetracker.desktop
```

**3. Paste this configuration into the file:**
*(Make sure to replace `YOUR_USERNAME` with your actual Pi username, e.g., `pi`, and adjust the path to wherever you saved `app.py`)*

```ini
[Desktop Entry]
Type=Application
Name=MouseTracker Pro
Comment=BioResearch Lab Telemetry Station
Exec=/usr/bin/python3 /home/YOUR_USERNAME/path_to_your_folder/app.py
Terminal=false
```

**4. Save and Exit:**
Press `Ctrl + O` to save, hit `Enter` to confirm, and then press `Ctrl + X` to exit nano.

### How it all comes together:
1. You plug in the Raspberry Pi 5.
2. It boots up, automatically logs into the desktop, reads that `.desktop` file, and instantly launches your Python script full-screen. 
3. The UI covers the standard desktop entirely (making it look like a dedicated medical appliance).
4. When the experiment is over, you tap **"POWER OFF"** on the screen. The app gracefully kills the camera thread and tells Linux to shut down safely.
5. Wait for the green light on the Pi to stop flashing, and you can safely unplug it.

Would you like to discuss how to log all of this real-time temperature and BPM data to a CSV file so you can pull it off the Pi with a USB stick later?