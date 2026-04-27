import re
from typing import Type, List, Optional
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import (
    QWidget, QLineEdit, QFrame, QToolButton, QColorDialog, QPushButton, QDialog, QLabel,
    QListWidget, QDialogButtonBox, QAbstractItemView, QVBoxLayout, QHBoxLayout, QMessageBox, QListWidgetItem
)


HEX_RE = re.compile(r'^#(?:[0-9A-Fa-f]{6})$')


class ColorRow(QWidget):

    def __init__(self, color: str):
        super().__init__()

        self.swatch = QFrame()
        self.swatch.setFixedSize(24, 24)
        self.swatch.setFrameShape(QFrame.Box)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText('#RRGGBB')
        self.edit.setFixedWidth(100)

        self.button_pick = QPushButton('Pick...')
        self.button_remove = QToolButton()
        self.button_remove.setText('✕')
        self.button_remove.setToolTip('Remove this color')

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)
        layout.addWidget(self.swatch)
        layout.addWidget(self.edit)
        layout.addWidget(self.button_pick)
        layout.addStretch()
        layout.addWidget(self.button_remove)

        self.edit.textChanged.connect(self._on_text_changed)
        self.button_pick.clicked.connect(self._pick_color)

        self.set_color(color)

    def set_color(self, color: str):
        color = self.normalize_hex(color)
        self.edit.setText(color)
        self._update_swatch(color)

    def get_color(self) -> str:
        return self.normalize_hex(self.edit.text())

    def is_valid(self) -> bool:
        return bool(HEX_RE.fullmatch(self.edit.text().strip()))

    def _on_text_changed(self, text: str):
        text = text.strip()
        if HEX_RE.fullmatch(text):
            self._update_swatch(text.upper())
            self.edit.setStyleSheet('')
        else:
            self.edit.setStyleSheet('QLineEdit { background: #fff3f3; }')

    def _pick_color(self):
        initial = QColor(self.edit.text()) if self.is_valid() else QColor('#1F77B4')
        color = QColorDialog.getColor(initial, self, 'Choose color')
        if color.isValid():
            self.set_color(color.name().upper())

    def _update_swatch(self, color: str):
        self.swatch.setStyleSheet(
            f'QFrame {{ background-color: {color}; border: 1px solid #666; }}'
        )

    @staticmethod
    def normalize_hex(text: str) -> str:
        text = text.strip()
        if not text:
            return '#000000'

        if not text.startswith('#'):
            text = '#' + text

        qcolor = QColor(text)
        if qcolor.isValid():
            return qcolor.name().upper()

        return text.upper()


class ColorDialog:

    MATPLOTLIB_DEFAULT_COLORS = [
        '#1F77B4',  # blue
        '#FF7F0E',  # orange
        '#2CA02C',  # green
        '#D62728',  # red
        '#9467BD',  # purple
        '#8C564B',  # brown
        '#E377C2',  # pink
        '#7F7F7F',  # gray
        '#BCBD22',  # yellow
        '#17BECF',  # cyan
    ]

    view: Type[QWidget]

    dialog: QDialog
    label: QLabel
    list_widget: QListWidget
    button_add: QPushButton
    button_duplicate: QPushButton
    button_up: QPushButton
    button_down: QPushButton
    button_preset: QPushButton
    button_box: QDialogButtonBox

    def __init__(self, view: Type[QWidget]):

        self.view = view
        self.dialog = QDialog(parent=self.view)
        self.dialog.setWindowTitle('Colors')
        self.dialog.resize(500, 500)

        self.label = QLabel('Colors used in plotting order')

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)

        self.button_add = QPushButton('Add')
        self.button_duplicate = QPushButton('Duplicate')
        self.button_up = QPushButton('Up')
        self.button_down = QPushButton('Down')
        self.button_preset = QPushButton('Reset to Default')

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        self._build_layout()
        self._connect_signals()
        self._load_colors(self.MATPLOTLIB_DEFAULT_COLORS)

    def __call__(self) -> Optional[List[str]]:
        result = self.dialog.exec()
        if result == QDialog.Accepted:
            return self.get_colors()
        return None

    def _build_layout(self):
        main = QVBoxLayout(self.dialog)
        main.addWidget(self.label)
        main.addWidget(self.list_widget)

        controls = QHBoxLayout()
        controls.addWidget(self.button_add)
        controls.addWidget(self.button_duplicate)
        controls.addWidget(self.button_up)
        controls.addWidget(self.button_down)
        controls.addStretch()
        controls.addWidget(self.button_preset)
        main.addLayout(controls)

        main.addWidget(self.button_box)

    def _connect_signals(self):
        self.button_add.clicked.connect(self.add_color)
        self.button_duplicate.clicked.connect(self.duplicate_selected)
        self.button_up.clicked.connect(self.move_up)
        self.button_down.clicked.connect(self.move_down)
        self.button_preset.clicked.connect(self.load_default_colors)

        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.dialog.reject)

    def _load_colors(self, colors: List[str]):
        self.list_widget.clear()
        for color in colors:
            self._add_row(color)

    def _create_row(self, color: str) -> ColorRow:
        row = ColorRow(color)
        row.button_remove.clicked.connect(lambda: self._remove_row_widget(row))
        return row

    def _add_row(self, color: str):
        row = self._create_row(color)

        item = QListWidgetItem()
        item.setSizeHint(row.sizeHint())

        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, row)

    def _insert_row(self, index: int, color: str):
        row = self._create_row(color)

        item = QListWidgetItem()
        item.setSizeHint(row.sizeHint())

        self.list_widget.insertItem(index, item)
        self.list_widget.setItemWidget(item, row)

    def _remove_row_widget(self, row_widget: ColorRow):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if self.list_widget.itemWidget(item) is row_widget:
                self.list_widget.takeItem(i)
                break

    def add_color(self):
        self._add_row(self.MATPLOTLIB_DEFAULT_COLORS[0])
        self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def duplicate_selected(self):
        row = self.list_widget.currentRow()
        if row < 0:
            return

        color = self._row_widget(row).get_color()
        self._insert_row(row + 1, color)
        self.list_widget.setCurrentRow(row + 1)

    def move_up(self):
        row = self.list_widget.currentRow()
        if row <= 0:
            return

        color = self._row_widget(row).get_color()
        self.list_widget.takeItem(row)
        self._insert_row(row - 1, color)
        self.list_widget.setCurrentRow(row - 1)

    def move_down(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= self.list_widget.count() - 1:
            return

        color = self._row_widget(row).get_color()
        self.list_widget.takeItem(row)
        self._insert_row(row + 1, color)
        self.list_widget.setCurrentRow(row + 1)

    def load_default_colors(self):
        self._load_colors(self.MATPLOTLIB_DEFAULT_COLORS)

    def get_colors(self) -> List[str]:
        colors = []
        for i in range(self.list_widget.count()):
            row = self._row_widget(i)
            colors.append(row.get_color())
        return colors

    def _validate(self) -> bool:
        if self.list_widget.count() == 0:
            QMessageBox.warning(
                self.dialog,
                'No colors',
                'Please specify at least one color.',
            )
            return False

        for i in range(self.list_widget.count()):
            row = self._row_widget(i)
            if not row.is_valid():
                QMessageBox.warning(
                    self.dialog,
                    'Invalid color',
                    f'Row {i + 1} does not contain a valid hex color (#RRGGBB).',
                )
                return False

        return True

    def _on_accept(self):
        if self._validate():
            self.dialog.accept()

    def _row_widget(self, row: int) -> ColorRow:
        item = self.list_widget.item(row)
        return self.list_widget.itemWidget(item)
