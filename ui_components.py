from PySide6.QtWidgets import QWidget, QLabel, QRubberBand
from PySide6.QtCore import Qt, Signal, QRect, QSize
from PySide6.QtGui import QPainter, QColor, QPen

class RangeSlider(QWidget):
    valueChanged = Signal(int, int)

    def __init__(self, minimum=0, maximum=100):
        super().__init__()
        self.setMinimumSize(100, 30)
        self.minimum = minimum
        self.maximum = maximum
        self._min_val = minimum
        self._max_val = maximum
        self.handle_radius = 8
        self.active_handle = None

    def setRange(self, min_val, max_val):
        self.minimum = min_val; self.maximum = max_val; self.update()

    def setValues(self, min_val, max_val):
        self._min_val = max(self.minimum, min_val)
        self._max_val = min(self.maximum, max_val)
        self.update()

    def val_to_pos(self, val):
        w = self.width() - 2 * self.handle_radius
        return self.handle_radius + int((val - self.minimum) / (self.maximum - self.minimum) * w) if self.maximum > self.minimum else 0

    def pos_to_val(self, pos):
        w = self.width() - 2 * self.handle_radius
        val = self.minimum + (pos - self.handle_radius) / w * (self.maximum - self.minimum)
        return max(self.minimum, min(self.maximum, int(val)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cy = self.height() // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#e1e9ee"))
        painter.drawRoundedRect(self.handle_radius, cy - 2, self.width() - 2 * self.handle_radius, 4, 2, 2)
        x1 = self.val_to_pos(self._min_val)
        x2 = self.val_to_pos(self._max_val)
        painter.setBrush(QColor("#005db5"))
        painter.drawRoundedRect(x1, cy - 2, x2 - x1, 4, 2, 2)
        painter.setBrush(QColor("white"))
        painter.setPen(QPen(QColor("#ccc"), 1))
        painter.drawEllipse(QRect(x1 - self.handle_radius, cy - self.handle_radius, self.handle_radius * 2, self.handle_radius * 2))
        painter.drawEllipse(QRect(x2 - self.handle_radius, cy - self.handle_radius, self.handle_radius * 2, self.handle_radius * 2))

    def mousePressEvent(self, event):
        pos = int(event.position().x())
        x1 = self.val_to_pos(self._min_val)
        x2 = self.val_to_pos(self._max_val)
        self.active_handle = 'min' if abs(pos - x1) < abs(pos - x2) else 'max'
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        val = self.pos_to_val(int(event.position().x()))
        if self.active_handle == 'min':
            self._min_val = min(val, self._max_val - 1)
        elif self.active_handle == 'max':
            self._max_val = max(val, self._min_val + 1)
        self.update()
        self.valueChanged.emit(self._min_val, self._max_val)

    def mouseReleaseEvent(self, event):
        self.active_handle = None

class InteractiveVideoLabel(QLabel):
    # Emits normalized coordinates (0.0 to 1.0) so it scales perfectly 
    roi_selected = Signal(float, float, float, float) 

    def __init__(self):
        super().__init__()
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = None
        self.selection_active = False

    def enable_selection(self):
        self.selection_active = True
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event):
        if not self.selection_active: return
        self.origin = event.position().toPoint()
        self.rubberBand.setGeometry(QRect(self.origin, QSize()))
        self.rubberBand.show()

    def mouseMoveEvent(self, event):
        if not self.selection_active or not self.origin: return
        self.rubberBand.setGeometry(QRect(self.origin, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event):
        if not self.selection_active or not self.origin: return
        self.rubberBand.hide()
        self.selection_active = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

        rect = self.rubberBand.geometry()
        if rect.width() < 10 or rect.height() < 10:
            return # Ignore accidental tiny clicks

        if not self.pixmap(): return
        
        # Map the screen click down to the actual video pixels
        pm_width = self.pixmap().width()
        pm_height = self.pixmap().height()
        offset_x = (self.width() - pm_width) / 2
        offset_y = (self.height() - pm_height) / 2

        x = max(0, rect.x() - offset_x)
        y = max(0, rect.y() - offset_y)

        # Normalize between 0 and 1
        nx = x / pm_width
        ny = y / pm_height
        nw = rect.width() / pm_width
        nh = rect.height() / pm_height

        self.roi_selected.emit(nx, ny, nw, nh)
