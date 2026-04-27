from typing import Type
from PyQt5.QtWidgets import QFileDialog, QWidget


class FileDialog:

    view: Type[QWidget]

    def __init__(self, view: Type[QWidget]):
        self.view = view


class OpenFileDialog(FileDialog):

    def __call__(self) -> str:
        d = QFileDialog(self.view)
        d.resize(1200, 800)
        d.setWindowTitle('Open')
        d.setNameFilters([
            'All Files (*.*)',
            'Excel Files (*.xlsx)',
            'Comma-Separated Files (*.csv)',
            'Tab-Delimited Files (*.tsv *.tab *.txt)',
        ])
        d.selectNameFilter('Excel Files (*.xlsx)')
        d.setOptions(QFileDialog.DontUseNativeDialog)
        d.setFileMode(QFileDialog.ExistingFile)  # only one existing file can be selected
        response = d.exec_()
        if response == QFileDialog.Accepted:
            selected = d.selectedFiles()
            if len(selected) > 0:
                return selected[0]
        return ''


class SaveAsFileDialog(FileDialog):

    def __call__(self, filename: str = '') -> str:
        d = QFileDialog(self.view)
        d.resize(1200, 800)
        d.setWindowTitle('Save As')
        d.selectFile(filename)
        d.setNameFilters([
            'All Files (*.*)',
            'Excel Files (*.xlsx)',
            'Comma-Separated Files (*.csv)',
            'Tab-Delimited Files (*.tsv *.tab *.txt)',
        ])
        d.selectNameFilter('Excel Files (*.xlsx)')
        d.setOptions(QFileDialog.DontUseNativeDialog)
        d.setAcceptMode(QFileDialog.AcceptSave)
        response = d.exec_()
        if response == QFileDialog.Accepted:
            selected = d.selectedFiles()
            if len(selected) > 0:
                return selected[0]
        return ''


class OpenDirectoryDialog(FileDialog):

    def __call__(self, caption: str) -> str:
        d = QFileDialog(self.view)
        d.resize(1200, 800)
        d.setWindowTitle(caption)
        d.setOptions(QFileDialog.DontUseNativeDialog)
        d.setFileMode(QFileDialog.DirectoryOnly)  # only one directory can be selected
        response = d.exec_()
        if response == QFileDialog.Accepted:
            selected = d.selectedFiles()
            if len(selected) > 0:
                return selected[0]
        return ''
