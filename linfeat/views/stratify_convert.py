import pandas as pd
from typing import List, Optional, Any, Dict, Type
from superqt import QRangeSlider
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QFontMetrics
from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QScrollArea, QLineEdit, QLabel
from .base import str_


class LabeledRangeSlider(QRangeSlider):
    """
    QRangeSlider is integer-based.
    Internally use a grid of 1000 steps, from 0 to 1000.

    The grid values are the values at the steps, defined by the minimum (float) and maximum (float).
    Grid values are strings, formatted to 3 significant digits.
    The reason to use strings is to avoid floating point precision issues.
    Also, string is more compatible with GUI elements like QComboBox.

    self.minimum() is the internal minimum value (0)
    self.maximum() is the internal maximum value (1000)
    self.value() is the list of positions (int) of the handles

    self.grid_values is the list of grid values (str)
    self.handlePressed emits the position (int) at which the handle is pressed
    """

    N_STEPS = 1000

    grid_values: List[str]

    handlePressed = pyqtSignal(int)

    def __init__(self, parent: QDialog):
        super().__init__(parent=parent, orientation=Qt.Horizontal)
        self.setRange(0, self.N_STEPS)
        self.set_cutoff_at_midpoint()
        self.setTickInterval(10)
        self.setTickPosition(QRangeSlider.TicksBelow)
        self.setMinimumHeight(70)

        self.set_grid_values(minimum=0.0, maximum=100.0)        

    def set_cutoff_at_midpoint(self):
        self.setValue([self.N_STEPS // 2])

    def set_grid_values(self, minimum: float, maximum: float):
        m = minimum
        M = maximum

        self.grid_values = [str(m)]  # the first should be exactly the minimum
        
        interval = (M - m) / self.N_STEPS
        for i in range(1, self.N_STEPS):
            value = m + i * interval
            self.grid_values.append(f'{value:.3g}')  # in the middle round to 3 significant digits
        
        self.grid_values.append(str(M))  # the last should be exactly the maximum

    def add_cutoff(self) -> List[str]:
        """
        Add a new handle at the midpoint of the largest available interval.

        Example:
            current handle positions: 20, 80

            intervals:
                0--20
                20--80
                80--100

            largest interval is 20--80, so new handle position is 50.
        """
        current_positions = list(self.value())
        boundaries = [0] + current_positions + [self.N_STEPS]

        best_left = None
        best_right = None
        best_width = -1

        for left, right in zip(boundaries[:-1], boundaries[1:]):
            width = right - left
            if width > best_width:
                best_width = width
                best_left = left
                best_right = right

        if best_width <= 1:
            return [self.grid_values[pos] for pos in self.value()]

        new_pos = int(round((best_left + best_right) / 2))
        current_positions.append(new_pos)
        current_positions = sorted(set(current_positions))

        self.setValue(current_positions)

        return [self.grid_values[pos] for pos in self.value()]

    def remove_cutoff(self, position: int):
        if len(self.value()) == 1:
            return  # do nothing if only one cutoff left

        current_positions = list(self.value())
        if position in current_positions:
            current_positions.remove(position)

        self.setValue(current_positions)

    def get_cutoffs(self) -> Dict[int, str]:
        ret = {}
        for position in self.value():
            ret[position] = self.grid_values[position]
        return ret

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        label_offset_y = 34
        y = self.height() // 2 + label_offset_y

        metrics = QFontMetrics(painter.font())
        for pos in self.value():
            text = self.grid_values[pos]

            text_width = metrics.horizontalAdvance(text)
            x = int(self._position_to_x(pos))
            text_x = x - text_width // 2
            text_x = max(2, min(text_x, self.width() - text_width - 2))

            painter.drawText(text_x, y, text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            position = self._nearest_handle_position(event.x())

            if position is not None:
                self.handlePressed.emit(position)

        super().mousePressEvent(event)

    def _nearest_handle_position(self, x) -> Optional[int]:
        handle_positions = list(self.value())

        min_distance = None
        at_position = None
        for pos in handle_positions:
            dist = abs(x - self._position_to_x(pos))
            if min_distance is None or dist < min_distance:
                min_distance = dist
                at_position = pos

        click_tolerance_px = 12
        if min_distance <= click_tolerance_px:
            return at_position

        return None

    def _position_to_x(self, position: int) -> int:
        m = self.minimum()  # should be 0
        M = self.maximum()  # should be self.N_STEPS

        if M == m:
            return 0

        margin = 14
        usable_width = self.width() - 2 * margin
        ratio = (position - m) / (M - m)

        return margin + ratio * usable_width


class StratifyDialog:

    view: Type[QWidget]
    dialog: QDialog
    slider: LabeledRangeSlider
    cutoff_combo: QComboBox

    cutoff_position_to_value: Dict[int, str]

    def __init__(self, view: Type[QWidget]):
        self.view = view
        self.dialog = QDialog(parent=self.view)
        self.dialog.setWindowTitle('Stratify')
        self.dialog.resize(600, 160)

        self.slider = LabeledRangeSlider(parent=self.dialog)
        self.slider.handlePressed.connect(self.on_handle_pressed)

        btn_add = QPushButton('Add Cutoff')
        btn_add.clicked.connect(self.add_cutoff)
        btn_remove = QPushButton('Remove Selected Cutoff')
        btn_remove.clicked.connect(self.remove_selected_cutoff)
        self.cutoff_combo = QComboBox(parent=self.dialog)

        self.slider.valueChanged.connect(self.update_combo_box_from_the_slider)

        edit_layout = QHBoxLayout()
        edit_layout.addWidget(btn_add)
        edit_layout.addWidget(self.cutoff_combo)
        edit_layout.addWidget(btn_remove)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.dialog.accept)
        button_box.rejected.connect(self.dialog.reject)

        layout = QVBoxLayout(self.dialog)
        layout.addWidget(self.slider)
        layout.addLayout(edit_layout)
        layout.addWidget(button_box)

        self.update_combo_box_from_the_slider()

    def __call__(self, minimum: float, maximum: float) -> Optional[List[float]]:
        self.slider.set_grid_values(minimum, maximum)
        self.slider.set_cutoff_at_midpoint()
        self.update_combo_box_from_the_slider()

        result = self.dialog.exec_()

        ret = None
        if result == QDialog.Accepted:
            ret = []
            for position, value in self.slider.get_cutoffs().items():
                ret.append(float(value))
        return ret

    def add_cutoff(self):
        self.slider.add_cutoff()
        self.update_combo_box_from_the_slider()

    def remove_selected_cutoff(self):
        current_value = self.cutoff_combo.currentText()

        # use current value to find the position
        position = None
        for position, value in self.cutoff_position_to_value.items():
            if value == current_value:
                break
        if position is None:
            return

        self.slider.remove_cutoff(position=position)
        self.update_combo_box_from_the_slider()

    def on_handle_pressed(self, position: int):
        """
        When the user clicks on a handle, update the combo box to the value of the handle.
        """
        value = self.cutoff_position_to_value.get(position, None)
        if value is None:
            return
        idx = self.cutoff_combo.findText(value)
        if idx != -1:
            self.cutoff_combo.setCurrentIndex(idx)

    def update_combo_box_from_the_slider(self):
        self.cutoff_position_to_value = self.slider.get_cutoffs()

        current_value = self.cutoff_combo.currentText()

        self.cutoff_combo.blockSignals(True)
        self.cutoff_combo.clear()

        for position, value in self.cutoff_position_to_value.items():
            self.cutoff_combo.addItem(value)

        idx = self.cutoff_combo.findText(current_value)
        if idx != -1:
            self.cutoff_combo.setCurrentIndex(idx)

        self.cutoff_combo.blockSignals(False)

        self.slider.update()  # force repaint so labels under handles stay updated


class ConvertDialog:

    view: Type[QWidget]

    dialog: QDialog
    form_layout: QFormLayout
    old_to_line_edits: Dict[Any, QLineEdit]

    scroll_area: QScrollArea
    scroll_widget: QWidget

    def __init__(self, view: Type[QWidget]):
        self.view = view

        self.dialog = QDialog(parent=self.view)
        self.dialog.setWindowTitle("Convert")
        self.dialog.resize(500, 600)

        main_layout = QVBoxLayout(self.dialog)

        # Scrollable form area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.scroll_widget = QWidget()
        self.form_layout = QFormLayout(self.scroll_widget)

        self.scroll_area.setWidget(self.scroll_widget)
        main_layout.addWidget(self.scroll_area)

        # OK / Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.dialog.accept)
        button_box.rejected.connect(self.dialog.reject)

        main_layout.addWidget(button_box)

        self.old_to_line_edits = {}

    def __call__(self, old_to_new: Dict[Any, Any]) -> Optional[Dict[Any, str]]:

        self.clear_form()

        for old, new in old_to_new.items():
            label = QLabel(str_(old))

            line_edit = QLineEdit()
            line_edit.setText(str_(new))

            self.form_layout.addRow(label, line_edit)
            self.old_to_line_edits[old] = line_edit

        result = self.dialog.exec_()

        if result != QDialog.Accepted:
            return None

        return {
            old: line_edit.text()
            for old, line_edit in self.old_to_line_edits.items()
        }

    def clear_form(self):
        while self.form_layout.rowCount() > 0:
            self.form_layout.removeRow(0)

        self.old_to_line_edits.clear()
