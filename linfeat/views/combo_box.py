import pandas as pd
from typing import List, Optional, Tuple, Type, Dict, Any
from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QScrollArea
from .base import str_


class EditRowDialog:

    WIDTH = 600
    HEIGHT = 600

    view: Type[QWidget]

    # statically initialized
    dialog: QDialog
    main_layout: QVBoxLayout
    form_layout: QFormLayout
    button_box: QDialogButtonBox

    # dynamically defined when calling __call__() method
    field_to_options: Dict[str, List[str]]
    field_to_combo_boxes: Dict[str, QComboBox]
    output_dict: Dict[str, str]

    def __init__(self, view: Type[QWidget]):
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


class SelectOutcomeDialog:

    view: Type[QWidget]

    dialog: QDialog
    combo_box: QComboBox

    def __init__(self, view: Type[QWidget]):
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


class FillMissingValuesDialog:

    BINARY_OPTIONS: List[str] = ['0', '1']
    CONTINUOUS_OPTIONS: List[str] = ['mean', 'median']
    CATEGORICAL_OPTIONS: List[str] = ['NA']

    view: Type[QWidget]

    dialog: QDialog
    layout: QFormLayout
    combo_binary: QComboBox
    combo_continuous: QComboBox
    combo_categorical: QComboBox
    button_box: QDialogButtonBox

    def __init__(self, view: Type[QWidget]):
        self.view = view

        self.dialog = QDialog(parent=self.view)
        self.dialog.setWindowTitle(' ')
        self.layout = QFormLayout(parent=self.dialog)

        self.combo_binary = QComboBox(parent=self.dialog)
        self.combo_binary.addItems(self.BINARY_OPTIONS)
        self.combo_binary.setEditable(False)  # values other than 0 or 1 are not allowed for binary variables
        self.combo_binary.setCurrentText(self.BINARY_OPTIONS[0])

        self.combo_continuous = QComboBox(parent=self.dialog)
        self.combo_continuous.addItems(self.CONTINUOUS_OPTIONS)
        self.combo_continuous.setEditable(True)  # user can input any numeric value for continuous variables
        self.combo_continuous.setCurrentText(self.CONTINUOUS_OPTIONS[0])

        self.combo_categorical = QComboBox(parent=self.dialog)
        self.combo_categorical.addItems(self.CATEGORICAL_OPTIONS)
        self.combo_categorical.setEditable(True)  # user can input any value (as string) for categorical variables
        self.combo_categorical.setCurrentText(self.CATEGORICAL_OPTIONS[0])

        self.layout.addRow('Binary:', self.combo_binary)
        self.layout.addRow('Continuous:', self.combo_continuous)
        self.layout.addRow('Categorical:', self.combo_categorical)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self.dialog,
        )
        self.button_box.accepted.connect(self.dialog.accept)
        self.button_box.rejected.connect(self.dialog.reject)
        self.layout.addWidget(self.button_box)

    def __call__(self) -> Optional[Tuple[str, str, str]]:
        if self.dialog.exec_() == QDialog.Accepted:
            return (
                self.combo_binary.currentText(),
                self.combo_continuous.currentText(),
                self.combo_categorical.currentText(),
            )
        return None
