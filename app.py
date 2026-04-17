import sys
import argparse
from PySide6.QtWidgets import QApplication
from dashboard import MouseTrackerDashboard

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameraIndex", type=int, default=0)
    parser.add_argument("--videoPath", type=str, default="")
    parser.add_argument("--test-mode", action="store_true", help="Run UI on PC with simulated hardware")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MouseTrackerDashboard(args)
    window.show()
    sys.exit(app.exec())