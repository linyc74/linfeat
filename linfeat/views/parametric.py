from typing import Optional, Dict, Type
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QAbstractItemView, QComboBox
from ..basic import CONTINUOUS


class SetParametricVariablesDialog:

    PARAMETRIC_OPTIONS = ['Parametric', 'Nonparametric']

    view: Type[QWidget]

    dialog: QDialog
    table: QTableWidget

    def __init__(self, view: Type[QWidget]):
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
        while self.table.rowCount() > 0:
            self.table.removeRow(0)

        packet = self.view.model.get_data_packet()
        variable_to_type = packet.column_to_type
        variable_to_parametric = packet.column_to_parametric
        forced_categorical_variables = packet.forced_categorical_columns

        self.table.setRowCount(len(variable_to_parametric))

        for row, (variable, parametric) in enumerate(variable_to_parametric.items()):
            name_item = QTableWidgetItem(variable)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)

            combo = QComboBox()
            combo.addItems(self.PARAMETRIC_OPTIONS)

            current_value = 'Parametric' if parametric else 'Nonparametric'
            combo.setCurrentText(current_value)

            if (not variable_to_type[variable] == CONTINUOUS) or (variable in forced_categorical_variables):
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
