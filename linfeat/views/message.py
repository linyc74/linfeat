from typing import Type
from PyQt5.QtWidgets import QMessageBox, QWidget


class Message:

    TITLE: str
    ICON: QMessageBox.Icon

    box: QMessageBox

    def __init__(self, view: Type[QWidget]):
        self.box = QMessageBox(parent=view)
        self.box.setWindowTitle(self.TITLE)
        self.box.setIcon(self.ICON)

    def __call__(self, msg: str):
        self.box.setText(msg)
        self.box.exec_()


class InfoMessage(Message):

    TITLE = 'Info'
    ICON = QMessageBox.Information


class ErrorMessage(Message):

    TITLE = 'Error'
    ICON = QMessageBox.Warning


class YesNoMessage(Message):

    TITLE = ' '
    ICON = QMessageBox.Question

    def __init__(self, view: Type[QWidget]):
        super().__init__(view=view)
        self.box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        self.box.setDefaultButton(QMessageBox.No)

    def __call__(self, msg: str) -> bool:
        self.box.setText(msg)
        return self.box.exec_() == QMessageBox.Yes


class UnsavedDataMessage(Message):

    TITLE = ' '
    ICON = QMessageBox.Warning

    def __init__(self, view: Type[QWidget]):
        super().__init__(view=view)
        self.box.setStandardButtons(QMessageBox.Save | QMessageBox.No | QMessageBox.Cancel)
        self.box.setDefaultButton(QMessageBox.Cancel)

    def __call__(self) -> int:
        self.box.setText('You have unsaved changes. Do you want to save them?')
        return self.box.exec_()  # QMessageBox.Save, QMessageBox.No, QMessageBox.Cancel are the returned integers
