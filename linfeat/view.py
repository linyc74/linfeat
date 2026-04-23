import re
import pandas as pd
from os.path import dirname, exists, expanduser
from typing import List, Optional, Any, Dict, Tuple

from superqt import QRangeSlider
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QObject
from PyQt5.QtGui import QIcon, QKeySequence, QColor, QPainter, QFontMetrics, QDropEvent
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QPushButton, QFileDialog, \
    QMessageBox, QGridLayout, QDialog, QFormLayout, QDialogButtonBox, QComboBox, QScrollArea, QLineEdit, \
    QShortcut, QAbstractItemView, QHBoxLayout, QListWidget, QListWidgetItem, QToolButton, QFrame, QLabel, \
    QColorDialog, QApplication

from .model import Model
from .basic import CONTINUOUS


class Table(QTableWidget):

    model: Model
    file_dropped = pyqtSignal(str)

    def __init__(self, model: Model):
        super().__init__()
        self.model = model

        self.setAcceptDrops(True)
        self.viewport().installEventFilter(self)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        
        self.refresh_table()

    def refresh_table(self):
        packet = self.model.get_data_packet()
        df = packet.df
        column_to_type = packet.column_to_type
        column_to_parametric = packet.column_to_parametric

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

        # convert to a pure numpy matrix of str to speed up
        df = df.map(str_).astype(str)
        matrix = df.to_numpy()

        # fill in values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # makes the item immutable, i.e. user cannot edit it
                self.setItem(i, j, item)

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

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self.viewport() and event.type() == QEvent.Drop:
            self._handle_url_drop(event)
            return True
        return super().eventFilter(obj, event)

    def dropEvent(self, event: QDropEvent) -> None:
        self._handle_url_drop(event)

    def _handle_url_drop(self, event: QDropEvent) -> None:
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            event.ignore()
            return

        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = expanduser(url.toLocalFile())
            if exists(path):
                self.file_dropped.emit(path)
                event.acceptProposedAction()
                return

        event.ignore()


class View(QWidget):

    TITLE = 'LinFeat'
    ICON_FILE = 'icon/logo.ico'
    WIDTH, HEIGHT = 1280, 768
    BUTTON_NAME_TO_LABEL = {
        'open': 'Open...',
        'save_as': 'Save As...',

        'undo': 'Undo',
        'sort_ascending': 'Sort (A to Z)',
        'delete_selected_rows': 'Delete Selected Samples',
        'add_new_row': 'Add New Sample',
        'edit_row': 'Edit Sample',
        'edit_cell': 'Edit',

        'redo': 'Redo',
        'sort_descending': 'Sort (Z to A)',
        'delete_selected_columns': 'Delete Selected Columns',
        'add_new_column': 'Add New Column',
        'rename_column': 'Rename Column',

        'stratify_convert': 'Stratify / Convert',
        'set_parametric_variables': 'Set Parametric Variables',
        'univariable_statistics': 'Univariable Statistics',
        'multivariable_regression': 'Multivariable Regression',
    
        # 'find': 'Find',  # hidden
    }
    BUTTON_NAME_TO_POSITION = {
        'open': (0, 0),
        'save_as': (1, 0),
        
        'undo': (0, 1),
        'sort_ascending': (1, 1),
        'delete_selected_rows': (2, 1),
        'add_new_row': (3, 1),
        'edit_row': (4, 1),
        'edit_cell': (5, 1),

        'redo': (0, 2),
        'sort_descending': (1, 2),
        'delete_selected_columns': (2, 2),
        'add_new_column': (3, 2),
        'rename_column': (4, 2),

        'stratify_convert': (0, 3),
        'set_parametric_variables': (1, 3),
        'univariable_statistics': (2, 3),
        'multivariable_regression': (3, 3),

        # 'find': (1, 4),
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
        self.dialog_set_parametric_variables = DialogSetParametricVariables(self)
        self.dialog_select_outcome = DialogSelectOutcome(self)
        self.dialog_colors = DialogColors(self)
        self.dialog_stratify = DialogStratify(self)
        self.dialog_convert = DialogConvert(self)
        self.dialog_new_column_name = DialogNewColumnName(self)
        self.dialog_rename_column = DialogRenameColumn(self)
    
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
        d.setNameFilters([
            'All Files (*.*)',
            'Excel Files (*.xlsx)',
            'Comma-Separated Files (*.csv)',
            'Tab-Delimited Files (*.tsv *.tab *.txt)',
        ])
        d.selectNameFilter('Excel Files (*.xlsx)')
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


class FileDialogOpenDirectory(FileDialog):

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
    HEIGHT = 600

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
        packet = self.view.model.get_data_packet()
        df = packet.df

        if len(df.columns) == 0:
            raise ValueError('Cannot add new row, the dataframe is empty.')

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
            self.form_layout.addRow(field, combo)
    
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


class DialogSetParametricVariables:

    PARAMETRIC_OPTIONS = ['Parametric', 'Nonparametric']

    view: View

    dialog: QDialog
    table: QTableWidget

    def __init__(self, view: View):
        self.view = view

        self.dialog = QDialog(parent=self.view)
        self.dialog.setWindowTitle('Set Parametric Variables')
        self.dialog.resize(500, 600)

        self.table = QTableWidget(self.dialog)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Variable', 'Parametric'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)

        btn_param = QPushButton('Set selected to Parametric')
        btn_nonparam = QPushButton('Set selected to Nonparametric')
        btn_ok = QPushButton('OK')
        btn_cancel = QPushButton('Cancel')

        btn_param.clicked.connect(lambda: self.set_selected_parametric('Parametric'))
        btn_nonparam.clicked.connect(lambda: self.set_selected_parametric('Nonparametric'))
        btn_ok.clicked.connect(self.dialog.accept)
        btn_cancel.clicked.connect(self.dialog.reject)

        button_row = QHBoxLayout()
        button_row.addWidget(btn_param)
        button_row.addWidget(btn_nonparam)
        button_row.addStretch()
        button_row.addWidget(btn_ok)
        button_row.addWidget(btn_cancel)

        layout = QVBoxLayout(self.dialog)
        layout.addWidget(self.table)
        layout.addLayout(button_row)

    def set_selected_parametric(self, value: str):
        selected_rows = sorted(set(index.row() for index in self.table.selectedIndexes()))
        for row in selected_rows:
            combo = self.table.cellWidget(row, 1)
            if combo is not None and combo.isEnabled():
                combo.setCurrentText(value)

    def __call__(self) -> Optional[Dict[str, str]]:
        self.render_table()
        if self.dialog.exec_() == QDialog.Accepted:
            return self.get_output_dict()
        else:
            return None

    def render_table(self):
        packet = self.view.model.get_data_packet()
        variable_to_type = packet.column_to_type
        variable_to_parametric = packet.column_to_parametric

        self.table.setRowCount(len(variable_to_parametric))

        for row, (variable, parametric) in enumerate(variable_to_parametric.items()):
            name_item = QTableWidgetItem(variable)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)

            combo = QComboBox()
            combo.addItems(self.PARAMETRIC_OPTIONS)

            current_value = 'Parametric' if parametric else 'Nonparametric'
            combo.setCurrentText(current_value)

            if not variable_to_type[variable] == CONTINUOUS:
                combo.setEnabled(False)
                combo.setCurrentText('Nonparametric')

            self.table.setCellWidget(row, 1, combo)

        self.table.resizeColumnsToContents()

    def get_output_dict(self) -> Optional[Dict[str, bool]]:
        ret = {}
        for row in range(self.table.rowCount()):
            variable = self.table.item(row, 0).text()
            parametric = self.table.cellWidget(row, 1).currentText()
            ret[variable] = parametric == 'Parametric'
        return ret


class DialogSelectOutcome:

    view: View

    dialog: QDialog
    combo_box: QComboBox

    def __init__(self, view: View):
        self.view = view

        self.dialog = QDialog(parent=self.view)
        self.dialog.setWindowTitle('Select Outcome')
        self.dialog.resize(500, 600)

        self.combo_box = QComboBox(parent=self.dialog)

        btn_ok = QPushButton('OK')
        btn_ok.clicked.connect(self.dialog.accept)
        btn_cancel = QPushButton('Cancel')
        btn_cancel.clicked.connect(self.dialog.reject)

        button_row = QHBoxLayout()
        button_row.addWidget(btn_ok)
        button_row.addWidget(btn_cancel)

        layout = QVBoxLayout(self.dialog)
        layout.addWidget(self.combo_box)
        layout.addLayout(button_row)

    def __call__(self) -> Optional[str]:
        self.combo_box.clear()

        packet = self.view.model.get_data_packet()
        variables = packet.df.columns.tolist()
        self.combo_box.addItems(variables)

        if self.dialog.exec_() == QDialog.Accepted:
            return self.combo_box.currentText()
        else:
            return None


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


class DialogColors:

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

    view: View

    dialog: QDialog
    label: QLabel
    list_widget: QListWidget
    button_add: QPushButton
    button_duplicate: QPushButton
    button_up: QPushButton
    button_down: QPushButton
    button_preset: QPushButton
    button_box: QDialogButtonBox

    def __init__(self, view: View):

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


class DialogStratify:

    view: View
    dialog: QDialog
    slider: LabeledRangeSlider
    cutoff_combo: QComboBox

    cutoff_position_to_value: Dict[int, str]

    def __init__(self, view: View):
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


class DialogConvert:

    view: View

    dialog: QDialog
    form_layout: QFormLayout
    old_to_line_edits: Dict[Any, QLineEdit]

    scroll_area: QScrollArea
    scroll_widget: QWidget

    def __init__(self, view: View):
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


class DialogNewColumnName(DialogLineEdits):

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


class DialogRenameColumn(DialogLineEdits):
    
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


def str_(value: Any) -> str:
    """
    Converts to str for GUI display
    """
    return '' if pd.isna(value) else str(value)
