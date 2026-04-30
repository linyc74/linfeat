import sys
from typing import List, Tuple
from os.path import dirname, exists, expanduser, normpath, join
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QObject
from PyQt5.QtGui import QIcon, QKeySequence, QDropEvent
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QPushButton, QGridLayout, QShortcut, QAbstractItemView
from .model import Model
from .views.base import str_
from .views.color import ColorDialog
from .views.parametric import SetParametricVariablesDialog
from .views.stratify_convert import StratifyDialog, ConvertDialog
from .views.message import InfoMessage, ErrorMessage, YesNoMessage
from .views.file import OpenFileDialog, SaveAsFileDialog, OpenDirectoryDialog
from .views.line_edit import FindDialog, EditCellDialog, NewColumnNameDialog, RenameColumnDialog
from .views.combo_box import EditRowDialog, SelectOutcomeDialog, FillMissingValuesDialog, NormalityTestDialog


def resource_path(*parts: str) -> str:
    """Return an absolute path to a packaged resource.

    Args:
        *parts: Path components under the resource root (e.g. ``('icon', 'logo.ico')``).

    Returns:
        Absolute filesystem path to the requested resource.

    Notes:
        - For Windows PyInstaller ``--onedir`` distributions, resources are expected to live next
          to the executable, e.g. ``<dist_dir>/icon/...``.
        - For macOS py2app bundles, resources live at
          ``<App>.app/Contents/Resources/...``.
        - For development runs, resources are expected to live in the repository root,
          e.g. ``<repo_root>/icon/...``.
    """
    if getattr(sys, 'frozen', False):
        exe_dir = dirname(sys.executable)

        # py2app: sys.executable is inside <App>.app/Contents/MacOS/
        # resources are located in <App>.app/Contents/Resources/
        if sys.platform == 'darwin':
            root_dir = normpath(join(exe_dir, '..', 'Resources'))
        else:
            root_dir = join(exe_dir, '_internal')
    else:
        # resources.py lives in <repo_root>/linfeat/resources.py
        root_dir = dirname(dirname(__file__))
    return join(root_dir, *parts)


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
        forced_categorical_columns = packet.forced_categorical_columns
        column_to_summary = packet.column_to_summary

        self.setRowCount(len(df.index))
        self.setColumnCount(len(df.columns))

        # render columns
        for i, column in enumerate(df.columns):
            item = QTableWidgetItem(column)
            if column in forced_categorical_columns:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                variable_type = 'forced_categorical'
            else:
                variable_type = column_to_type[column]
            summary = column_to_summary.get(column, '')
            png = f'{variable_type}.png'
            icon_file = resource_path('icon', png)
            item.setIcon(QIcon(icon_file))
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            item.setToolTip(summary)
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
    WIDTH, HEIGHT = 1280, 768
    BUTTON_NAME_TO_LABEL = {
        'open': 'Open...',
        'save_as': 'Save As...',

        'undo': 'Undo',
        'redo': 'Redo',
        'sort_ascending': 'Sort (A to Z)',
        'sort_descending': 'Sort (Z to A)',

        'delete_rows': 'Delete Samples',
        'add_new_row': 'Add New Sample',
        'edit_row': 'Edit Sample',
        'edit_cell': 'Edit',
        
        'delete_columns': 'Delete Columns',
        'add_new_column': 'Add New Column',
        'rename_column': 'Rename Column',

        'stratify_convert': 'Stratify / Convert',
        'force_categorical': 'Set as Categorical',
        'unforce_categorical': 'Unset Categorical',
        'fill_missing_values': 'Fill Missing Values',

        'set_parametric_variables': 'Set Parametric Variables',
        'normality_test': 'Normality Test',
        'univariable_statistics': 'Univariable Statistics',
        'multivariable_regression': 'Multivariable Regression',
    
        # 'find': 'Find',  # hidden
    }
    BUTTON_NAME_TO_POSITION = {
        'open': (0, 0),
        'save_as': (1, 0),
        
        'undo': (0, 1),
        'redo': (1, 1),
        'sort_ascending': (2, 1),
        'sort_descending': (3, 1),

        'delete_rows': (0, 2),
        'add_new_row': (1, 2),
        'edit_row': (2, 2),
        'edit_cell': (3, 2),

        'delete_columns': (0, 3),
        'add_new_column': (1, 3),
        'rename_column': (2, 3),

        'stratify_convert': (0, 4),
        'force_categorical': (1, 4),
        'unforce_categorical': (2, 4),
        'fill_missing_values': (3, 4),

        'set_parametric_variables': (0, 5),
        'normality_test': (1, 5),
        'univariable_statistics': (2, 5),
        'multivariable_regression': (3, 5),

        # 'find': (0, 6),
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
        icon_file = resource_path('icon', 'logo.ico')
        self.setWindowIcon(QIcon(icon_file))
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
        self.open_file_dialog = OpenFileDialog(self)
        self.save_as_file_dialog = SaveAsFileDialog(self)
        self.open_directory_dialog = OpenDirectoryDialog(self)

        self.info_message = InfoMessage(self)
        self.error_message = ErrorMessage(self)
        self.yes_no_message = YesNoMessage(self)

        self.set_parametric_variables_dialog = SetParametricVariablesDialog(self)

        self.edit_row_dialog = EditRowDialog(self)
        self.select_outcome_dialog = SelectOutcomeDialog(self)
        self.fill_missing_values_dialog = FillMissingValuesDialog(self)
        self.normality_test_dialog = NormalityTestDialog(self)
        
        self.color_dialog = ColorDialog(self)

        self.stratify_dialog = StratifyDialog(self)
        self.convert_dialog = ConvertDialog(self)

        self.edit_cell_dialog = EditCellDialog(self)
        self.find_dialog = FindDialog(self)
        self.new_column_name_dialog = NewColumnNameDialog(self)
        self.rename_column_dialog = RenameColumnDialog(self)

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
