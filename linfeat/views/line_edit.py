import pandas as pd
from typing import List, Optional, Any, Type
from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QLineEdit, QWidget
from .base import str_


class LineEdit:

    LINE_TITLES: List[str]
    LINE_DEFAULTS: List[str]

    view: Type[QWidget]

    dialog: QDialog
    layout: QFormLayout
    line_edits: List[QLineEdit]
    button_box: QDialogButtonBox

    def __init__(self, view: Type[QWidget]):
        self.view = view
        self.__init__dialog()
        self.__init__layout()
        self.__init__line_edits()
        self.__init__button_box()

    def __init__dialog(self):
        self.dialog = QDialog(parent=self.view)
        self.dialog.setWindowTitle(' ')

    def __init__layout(self):
        self.layout = QFormLayout(parent=self.dialog)

    def __init__line_edits(self):
        self.line_edits = []
        for title, default in zip(self.LINE_TITLES, self.LINE_DEFAULTS):
            line_edit = QLineEdit(default, parent=self.dialog)
            self.line_edits.append(line_edit)
            self.layout.addRow(title, line_edit)

    def __init__button_box(self):
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self.dialog)
        self.button_box.accepted.connect(self.dialog.accept)
        self.button_box.rejected.connect(self.dialog.reject)
        self.layout.addWidget(self.button_box)


class FindDialog(LineEdit):

    LINE_TITLES = [
        'Find:',
    ]
    LINE_DEFAULTS = [
        '',
    ]

    def __call__(self) -> Optional[str]:
        if self.dialog.exec_() == QDialog.Accepted:
            return self.line_edits[0].text()
        else:
            return None


class EditCellDialog(LineEdit):

    LINE_TITLES = [
        'Edit Cell:',
    ]
    LINE_DEFAULTS = [
        '',
    ]

    def __call__(self, value: Any) -> Optional[str]:
        self.line_edits[0].setText(str_(value))
        if self.dialog.exec_() == QDialog.Accepted:
            return self.line_edits[0].text()
        else:
            return None


class NewColumnNameDialog(LineEdit):

    LINE_TITLES = [
        'New Column:',
    ]
    LINE_DEFAULTS = [
        '',
    ]

    def __call__(self, name: str) -> Optional[str]:
        self.line_edits[0].setText(name)
        if self.dialog.exec_() == QDialog.Accepted:
            return self.line_edits[0].text()
        return None


class RenameColumnDialog(LineEdit):
    
    LINE_TITLES = [
        'Rename Column:',
    ]
    LINE_DEFAULTS = [
        '',
    ]

    def __call__(self, name: str) -> Optional[str]:
        self.line_edits[0].setText(name)
        if self.dialog.exec_() == QDialog.Accepted:
            return self.line_edits[0].text()
        return None

