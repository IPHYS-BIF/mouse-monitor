from PySide6.QtWidgets import QWidget, QLabel, QRubberBand, QSizePolicy
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
        
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        size_policy.setHeightForWidth(True)
        self.setSizePolicy(size_policy)

    def heightForWidth(self, width):
        if self.pixmap() and not self.pixmap().isNull():
            return int(width * self.pixmap().height() / max(1, self.pixmap().width()))
        return int(width * 3 / 4) # Default 4:3

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
        
        # Get original pixmap dimensions (frame dimensions)
        pm_width = self.pixmap().width()
        pm_height = self.pixmap().height()
        
        # Calculate displayed pixmap size (accounting for KeepAspectRatio scaling)
        label_width = self.width()
        label_height = self.height()
        
        # Calculate scaling factors
        scale_x = label_width / pm_width
        scale_y = label_height / pm_height
        scale = min(scale_x, scale_y)  # Keep aspect ratio
        
        # Calculate actual displayed dimensions
        displayed_width = pm_width * scale
        displayed_height = pm_height * scale
        
        # Calculate offset (centered within label)
        offset_x = (label_width - displayed_width) / 2.0
        offset_y = (label_height - displayed_height) / 2.0

        # Convert screen coordinates to displayed pixmap coordinates
        x = (rect.x() - offset_x) / scale
        y = (rect.y() - offset_y) / scale
        w = rect.width() / scale
        h = rect.height() / scale

        # Clamp to frame bounds and normalize to [0, 1]
        x = max(0.0, min(x / pm_width, 1.0))
        y = max(0.0, min(y / pm_height, 1.0))
        w = max(0.0, min(w / pm_width, 1.0))
        h = max(0.0, min(h / pm_height, 1.0))

        self.roi_selected.emit(x, y, w, h)
