import pandas as pd
from os.path import dirname, exists
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QPushButton, QFileDialog, \
    QMessageBox, QGridLayout, QDialog, QFormLayout, QDialogButtonBox, QComboBox, QScrollArea, QLineEdit, \
    QShortcut
from typing import List, Optional, Any, Dict, Tuple
from .model import Model


class Table(QTableWidget):

    model: Model

    def __init__(self, model: Model):
        super().__init__()
        self.model = model
        self.refresh_table()

    def refresh_table(self):
        df, column_to_type = self.model.get_data_packet()

        self.setRowCount(len(df.index))
        self.setColumnCount(len(df.columns))

        # render columns
        for i, column in enumerate(df.columns):
            item = QTableWidgetItem(column)
            variable_type = column_to_type[column]
            icon_path = f'icon/{variable_type}.png'
            item.setIcon(QIcon(icon_path))
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.setHorizontalHeaderItem(i, item)

        # fill in values
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                value = df.iloc[i, j]
                item = QTableWidgetItem(str_(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # makes the item immutable, i.e. user cannot edit it
                self.setItem(i, j, item)

        self.resizeColumnsToContents()

    def get_selected_rows(self) -> List[int]:
        ret = []
        for item in self.selectedItems():
            ith_row = item.row()
            if ith_row not in ret:
                ret.append(ith_row)
        return ret

    def get_selected_columns(self) -> List[str]:
        ret = []
        for item in self.selectedItems():
            ith_col = item.column()
            column = self.horizontalHeaderItem(ith_col).text()
            if column not in ret:
                ret.append(column)
        return ret

    def get_selected_cells(self) -> List[Tuple[int, str]]:
        ret = []
        for item in self.selectedItems():
            ith_row = item.row()
            ith_col = item.column()
            column = self.horizontalHeaderItem(ith_col).text()
            ret.append((ith_row, column))
        return ret

    def select_cell(self, index: int, column: str):
        ith_row = index
        columns = [self.horizontalHeaderItem(i).text() for i in range(self.columnCount())]
        ith_col = columns.index(column)
        self.setCurrentCell(ith_row, ith_col)


class View(QWidget):

    TITLE = 'LinFeat'
    ICON_FILE = 'icon/logo.ico'
    WIDTH, HEIGHT = 1280, 768
    BUTTON_NAME_TO_LABEL = {
        'open': 'Open...',
        'save_as': 'Save As...',
        'add_new_row': 'Add New Row',
        'edit_row': 'Edit Row',
        'edit_cell': 'Edit Cell',

        'undo': 'Undo',
        'redo': 'Redo',
        'find': 'Find',
        'sort_ascending': 'Sort (A to Z)',
        'sort_descending': 'Sort (Z to A)',
        'delete_selected_rows': 'Delete Selected Rows',
        'delete_selected_columns': 'Delete Selected Columns',

        'univariable_statistics': 'Univariable Statistics',
        'multivariable_regression': 'Multivariable Regression',
    }
    BUTTON_NAME_TO_POSITION = {
        'open': (0, 0),
        'save_as': (1, 0),
        'add_new_row': (2, 0),
        'edit_row': (3, 0),
        'edit_cell': (4, 0),

        'undo': (0, 1),
        'redo': (0, 2),
        'sort_ascending': (1, 1),
        'sort_descending': (1, 2),
        'delete_selected_rows': (2, 1),
        'delete_selected_columns': (2, 2),
        'find': (3, 1),

        'univariable_statistics': (0, 3),
        'multivariable_regression': (1, 3),
    }
    SHORTCUT_NAME_TO_KEY_SEQUENCE = {
        'control_s': 'Ctrl+S',
        'control_f': 'Ctrl+F',
        'control_z': 'Ctrl+Z',
        'control_y': 'Ctrl+Y',
    }

    model: Model
    vertical_layout: QVBoxLayout
    table: Table
    button_grid: QGridLayout

    def __init__(self, model: Model):
        super().__init__()
        self.model = model

        self.setWindowTitle(f'{self.TITLE}')
        self.setWindowIcon(QIcon(f'{dirname(dirname(__file__))}/{self.ICON_FILE}'))
        self.resize(self.WIDTH, self.HEIGHT)

        self.__init__vertical_layout()
        self.__init__main_table()
        self.__init__buttons()
        self.__init__shortcuts()
        self.__init__methods()
        self.show()

    def __init__vertical_layout(self):
        self.vertical_layout = QVBoxLayout()
        self.setLayout(self.vertical_layout)

    def __init__main_table(self):
        self.table = Table(self.model)
        self.vertical_layout.addWidget(self.table)

    def __init__buttons(self):
        self.button_grid = QGridLayout()
        self.vertical_layout.addLayout(self.button_grid)

        for name, label in self.BUTTON_NAME_TO_LABEL.items():
            setattr(self, f'button_{name}', QPushButton(label))
            button = getattr(self, f'button_{name}')
            pos = self.BUTTON_NAME_TO_POSITION[name]
            self.button_grid.addWidget(button, *pos)

    def __init__shortcuts(self):
        for name, key_sequence in self.SHORTCUT_NAME_TO_KEY_SEQUENCE.items():
            shortcut = QShortcut(QKeySequence(key_sequence), self)
            setattr(self, f'shortcut_{name}', shortcut)

    def __init__methods(self):
        self.file_dialog_open_table = FileDialogOpenTable(self)
        self.file_dialog_save_table = FileDialogSaveTable(self)
        self.file_dialog_open_directory = FileDialogOpenDirectory(self)
        self.message_box_info = MessageBoxInfo(self)
        self.message_box_error = MessageBoxError(self)
        self.message_box_yes_no = MessageBoxYesNo(self)
        self.dialog_edit_row = DialogEditRow(self)
        self.dialog_edit_cell = DialogEditCell(self)
        self.dialog_find = DialogFind(self)

    def refresh_table(self):
        self.table.refresh_table()

    def get_selected_rows(self) -> List[int]:
        return self.table.get_selected_rows()

    def get_selected_columns(self) -> List[str]:
        return self.table.get_selected_columns()

    def get_selected_cells(self) -> List[Tuple[int, str]]:
        return self.table.get_selected_cells()

    def select_cell(self, index: int, column: str):
        self.table.select_cell(index=index, column=column)


class FileDialog:

    view: View

    def __init__(self, view: View):
        self.view = view


class FileDialogOpenTable(FileDialog):

    def __call__(self) -> str:
        d = QFileDialog(self.view)
        d.resize(1200, 800)
        d.setWindowTitle('Open')
        d.setNameFilter('All Files (*.*);;CSV files (*.csv);;Excel files (*.xlsx)')
        d.selectNameFilter('CSV files (*.csv)')
        d.setOptions(QFileDialog.DontUseNativeDialog)
        d.setFileMode(QFileDialog.ExistingFile)  # only one existing file can be selected
        d.exec_()
        selected = d.selectedFiles()
        return selected[0] if len(selected) > 0 else ''


class FileDialogSaveTable(FileDialog):

    def __call__(self, filename: str = '') -> str:
        d = QFileDialog(self.view)
        d.resize(1200, 800)
        d.setWindowTitle('Save As')
        d.selectFile(filename)
        d.setNameFilter('All Files (*.*);;CSV files (*.csv);;Excel files (*.xlsx)')
        d.selectNameFilter('CSV files (*.csv)')
        d.setOptions(QFileDialog.DontUseNativeDialog)
        d.setAcceptMode(QFileDialog.AcceptSave)
        d.exec_()
        selected = d.selectedFiles()
        return selected[0] if len(selected) > 0 else ''


class FileDialogOpenDirectory(FileDialog):

    def __call__(self, caption: str) -> str:
        d = QFileDialog(self.view)
        d.resize(1200, 800)
        d.setWindowTitle(caption)
        d.setOptions(QFileDialog.DontUseNativeDialog)
        d.setFileMode(QFileDialog.DirectoryOnly)  # only one directory can be selected
        d.exec_()
        selected = d.selectedFiles()
        return selected[0] if len(selected) > 0 else ''


class MessageBox:

    TITLE: str
    ICON: QMessageBox.Icon

    box: QMessageBox

    def __init__(self, view: View):
        self.box = QMessageBox(parent=view)
        self.box.setWindowTitle(self.TITLE)
        self.box.setIcon(self.ICON)

    def __call__(self, msg: str):
        self.box.setText(msg)
        self.box.exec_()


class MessageBoxInfo(MessageBox):

    TITLE = 'Info'
    ICON = QMessageBox.Information


class MessageBoxError(MessageBox):

    TITLE = 'Error'
    ICON = QMessageBox.Warning


class MessageBoxYesNo(MessageBox):

    TITLE = ' '
    ICON = QMessageBox.Question

    def __init__(self, view: View):
        super().__init__(view=view)
        self.box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        self.box.setDefaultButton(QMessageBox.No)

    def __call__(self, msg: str) -> bool:
        self.box.setText(msg)
        return self.box.exec_() == QMessageBox.Yes


class DialogEditRow:

    WIDTH = 600
    HEIGHT =800

    view: QWidget

    # statically initialized
    dialog: QDialog
    main_layout: QVBoxLayout
    form_layout: QFormLayout
    button_box: QDialogButtonBox

    # dynamically defined when calling __call__() method
    field_to_options: Dict[str, List[str]]
    field_to_combo_boxes: Dict[str, QComboBox]
    output_dict: Dict[str, str]

    def __init__(self, view: View):
        self.view = view

        self.init_dialog()
        self.init_layout()
        self.init_button_box()
        
        self.field_to_options = {}
        self.field_to_combo_boxes = {}
        self.output_dict = {}

    def init_dialog(self):
        self.dialog = QDialog(parent=self.view)
        self.dialog.setWindowTitle(' ')
        self.dialog.resize(self.WIDTH, self.HEIGHT)

    def init_layout(self):
        self.main_layout = QVBoxLayout(self.dialog)

        # Create a ScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)  # Important: makes the inner widget resize with the scroll area

        # Create a QWidget for the scroll area content
        scroll_contents = QWidget(scroll)
        self.form_layout = QFormLayout(scroll_contents)

        # Set the scroll area's widget to be the QWidget with all items
        scroll.setWidget(scroll_contents)

        # Add ScrollArea to the main layout
        self.main_layout.addWidget(scroll)

    def init_button_box(self):
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self.dialog)
        self.button_box.accepted.connect(self.dialog.accept)
        self.button_box.rejected.connect(self.dialog.reject)
        self.main_layout.addWidget(self.button_box)

    def __call__(
            self,
            attributes: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:

        self.set_field_to_options()
        self.set_field_to_combo_boxes()
        self.set_combo_box_text(attributes=attributes)

        if self.dialog.exec_() == QDialog.Accepted:
            return self.get_output_dict()
        else:
            return None

    def set_field_to_options(self):
        self.field_to_options = {}
        
        # get all possible values for each field from the model
        df, column_to_type = self.view.model.get_data_packet()
        for field in df.columns:
            unique_values = df[field].dropna().unique()
            self.field_to_options[field] = [str_(value) for value in unique_values]

    def set_field_to_combo_boxes(self):
        self.field_to_combo_boxes = {}

        # remove all combo boxes from the form layout
        while self.form_layout.rowCount() > 0:
            self.form_layout.removeRow(0)

        for field, options in self.field_to_options.items():
            combo = QComboBox(parent=self.dialog)
            combo.addItems(options)
            combo.setEditable(True)
            self.field_to_combo_boxes[field] = combo
            self.form_layout.addRow(to_title(field), combo)
    
    def set_combo_box_text(self, attributes: Optional[Dict[str, Any]] = None):
        if attributes is None:  # set all to empty string
            for combo in self.field_to_combo_boxes.values():
                combo.setCurrentText('')
        else:
            for field, value in attributes.items():
                if field in self.field_to_combo_boxes.keys():
                    self.field_to_combo_boxes[field].setCurrentText(str_(value))     

    def get_output_dict(self) -> Dict[str, str]:
        return {
            field: combo.currentText()
            for field, combo in self.field_to_combo_boxes.items()
        } 


class DialogLineEdits:

    LINE_TITLES: List[str]
    LINE_DEFAULTS: List[str]

    view: View

    dialog: QDialog
    layout: QFormLayout
    line_edits: List[QLineEdit]
    button_box: QDialogButtonBox

    def __init__(self, view: View):
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


class DialogFind(DialogLineEdits):

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


class DialogEditCell(DialogLineEdits):

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


def str_(value: Any) -> str:
    """
    Converts to str for GUI display
    """
    return '' if pd.isna(value) else str(value)


def to_title(s: str) -> str:
    """
    'Title of good and evil' -> 'Title of Good and Evil'
    'title_of_good_and_evil' -> 'Title of Good and Evil'
    """
    skip = [
        'of',
        'and',
        'or',
        'on',
        'after',
        'about',
        'mAb',  # monoclonal antibody
    ]

    words = s.replace('_', ' ').split(' ')

    for i, word in enumerate(words):
        if word in skip:
            continue
        words[i] = word[0].upper() + word[1:]  # only capitalize the first letter

    return ' '.join(words)
