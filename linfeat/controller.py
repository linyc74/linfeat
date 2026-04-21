import shutil
import pandas as pd
from typing import Dict, Optional
from .view import View
from .model import Model
from .basic import CONTINUOUS


class Controller:

    model: Model
    view: View

    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view
        self.__init_actions()
        self.__connect_button_actions()
        self.__connect_short_actions()

        # for testing
        self.model.open(file='~/Desktop/small.csv')
        self.view.refresh_table()

    def __init_actions(self):
        self.action_open = ActionOpen(self)
        self.action_save_as = ActionSaveAs(self)
        self.action_add_new_row = ActionAddNewRow(self)
        self.action_edit_row = ActionEditRow(self)
        self.action_edit_cell = ActionEditCell(self)
        self.action_undo = ActionUndo(self)
        self.action_redo = ActionRedo(self)
        self.action_find = ActionFind(self)
        self.action_rename_column = ActionRenameColumn(self)
        self.action_sort_ascending = ActionSortAscending(self)
        self.action_sort_descending = ActionSortDescending(self)
        self.action_delete_selected_rows = ActionDeleteSelectedRows(self)
        self.action_delete_selected_columns = ActionDeleteSelectedColumns(self)
        self.action_control_s = ActionSaveAs(self)
        self.action_control_f = ActionFind(self)
        self.action_control_z = ActionUndo(self)
        self.action_control_y = ActionRedo(self)
        self.action_set_parametric_variables = ActionSetParametricVariables(self)
        self.action_univariable_statistics = ActionUnivariableStatistics(self)
        self.action_multivariable_regression = ActionMultivariableRegression(self)
        self.action_stratify_convert = ActionStratifyConvert(self)

    def __connect_button_actions(self):
        for name in self.view.BUTTON_NAME_TO_LABEL.keys():
            button = getattr(self.view, f'button_{name}')
            method = getattr(self, f'action_{name}', None)
            if method is not None:
                button.clicked.connect(method)
            else:
                print(f'WARNING: Controller method "action_{name}" does not exist for the button "{name}"')

    def __connect_short_actions(self):
        for name in self.view.SHORTCUT_NAME_TO_KEY_SEQUENCE.keys():
            shortcut = getattr(self.view, f'shortcut_{name}')
            method = getattr(self, f'action_{name}', None)
            if method is not None:
                shortcut.activated.connect(method)
            else:
                print(f'WARNING: Controller method "action_{name}" does not exist for the shortcut "{name}"')


class Action:

    model: Model
    view: View

    def __init__(self, controller: Controller):
        self.model = controller.model
        self.view = controller.view

    def __call__(self):
        try:
            self.action()
        except Exception as e:
            self.view.message_box_error(msg=repr(e))
        
    def action(self) -> None:
        raise NotImplementedError('Action method not implemented')


class ActionOpen(Action):

    def action(self):
        file = self.view.file_dialog_open_table()
        if file == '':
            return
        self.model.open(file=file)
        self.view.refresh_table()


class ActionSaveAs(Action):

    def action(self):
        file = self.view.file_dialog_save_table(filename='New Table.xlsx')
        if file == '':
            return
        self.model.save(file=file)


class ActionFind(Action):

    def action(self):
        text = self.view.dialog_find()
        if text is None:
            return

        selected_cells = self.view.get_selected_cells()
        start = None if len(selected_cells) == 0 else selected_cells[0]

        found_cell = self.model.find(text=text, start=start)

        if found_cell is None:
            self.view.message_box_info(msg='Couldn\'t find what you were looking for')
            return

        index, column = found_cell
        self.view.select_cell(index=index, column=column)


class ActionSort(Action):

    ASCENDING: bool

    def action(self):
        columns = self.view.get_selected_columns()
        if len(columns) == 0:
            self.view.message_box_error(msg='Please select a column')
        elif len(columns) == 1:
            self.model.sort_dataframe(by=columns[0], ascending=self.ASCENDING)
            self.view.refresh_table()
        else:
            self.view.message_box_error(msg='Please select only one column')


class ActionSortAscending(ActionSort):

    ASCENDING = True


class ActionSortDescending(ActionSort):

    ASCENDING = False


class ActionDeleteSelectedRows(Action):

    def action(self):
        rows = self.view.get_selected_rows()
        if len(rows) == 0:
            return
        if self.view.message_box_yes_no(msg='Are you sure you want to delete the selected rows?'):
            self.model.drop(rows=rows)
            self.view.refresh_table()


class ActionDeleteSelectedColumns(Action):

    def action(self):
        columns = self.view.get_selected_columns()
        if len(columns) == 0:
            return
        if self.view.message_box_yes_no(msg='Are you sure you want to delete the selected columns?'):
            self.model.drop(columns=columns)
            self.view.refresh_table()


class ActionAddNewRow(Action):

    def action(self):
        attributes = self.view.dialog_edit_row(attributes=None)
        if attributes is None:
            return
        
        self.model.append_row(attributes=attributes)
        self.view.refresh_table()

        last_row = len(self.model.dataframe.index) - 1
        first_column = self.model.dataframe.columns[0]
        self.view.select_cell(index=last_row, column=first_column)


class ActionEditRow(Action):

    def action(self):
        rows = self.view.get_selected_rows()

        if len(rows) == 0:
            self.view.message_box_error(msg='Please select a row')
            return
        elif len(rows) > 1:
            self.view.message_box_error(msg='Please select only one row')
            return

        row = rows[0]  # only one row is selected

        attributes = self.model.get_row(row=row)

        attributes = self.view.dialog_edit_row(attributes=attributes)
        if attributes is None:
            return
        self.model.update_row(row=row, attributes=attributes)

        self.view.refresh_table()


class ActionEditCell(Action):

    def action(self):
        cells = self.view.get_selected_cells()

        if len(cells) == 0:
            self.view.message_box_error('Please select a cell')
            return
        elif len(cells) > 1:
            self.view.message_box_error('Please select only one cell')
            return

        row, column = cells[0]
        value = self.model.get_value(row=row, column=column)

        new_value = self.view.dialog_edit_cell(value=value)

        if new_value is None:
            return

        self.model.update_cell(row=row, column=column, value=new_value)
        self.view.refresh_table()


class ActionUndo(Action):

    def action(self):
        self.model.undo()
        self.view.refresh_table()


class ActionRedo(Action):

    def action(self):
        self.model.redo()
        self.view.refresh_table()


class ActionSetParametricVariables(Action):

    def action(self):
        variable_to_parametric = self.view.dialog_set_parametric_variables()
        if variable_to_parametric is None:
            return
        for variable, parametric in variable_to_parametric.items():
            self.model.set_column_parametric(column=variable, parametric=parametric)


class ActionUnivariableStatistics(Action):

    def action(self):
        outdir = self.view.file_dialog_open_directory(caption='Select Output Directory')
        if outdir == '':
            return
        outcome = self.view.dialog_select_outcome()
        if outcome is None:
            return
        colors = self.view.dialog_colors()
        if colors is None:
            return
        self.model.univariable_statistics(outdir=outdir, outcome=outcome, colors=colors)
        self.view.message_box_info(msg='Univariable statistics completed')


class ActionMultivariableRegression(Action):

    def action(self):
        outdir = self.view.file_dialog_open_directory(caption='Select Output Directory')
        if outdir == '':
            return
        outcome = self.view.dialog_select_outcome()
        if outcome is None:
            return
        self.model.multivariable_regression(outdir=outdir, outcome=outcome)
        self.view.message_box_info(msg='Multivariable regression completed')


class ActionStratifyConvert(Action):

    def action(self):
        columns = self.view.get_selected_columns()
        if len(columns) == 0:
            self.view.message_box_error(msg='Please select a column')
            return
        elif len(columns) > 1:
            self.view.message_box_error(msg='Please select only one column')
            return
        column = columns[0]

        packet = self.model.get_data_packet()
        df = packet.df
        if packet.column_to_type[column] == CONTINUOUS:
            self.stratify_continous_variable(df=df, column=column)
        else:
            self.convert_categorical_variable(df=df, column=column)

    def stratify_continous_variable(self, df: pd.DataFrame, column: str):
        cutoffs = self.view.dialog_stratify(minimum=df[column].min(), maximum=df[column].max())
        if cutoffs is None:
            return

        cutoffs = [df[column].min()] + cutoffs + [df[column].max()]
        old_to_new = {}
        intervals = []
        for i in range(len(cutoffs) - 1):
            a = cutoffs[i]
            b = cutoffs[i + 1]
            intervals.append((a, b))
            old_to_new[f'{a} - {b}'] = i + 1

        old_to_new = self.view.dialog_convert(old_to_new=old_to_new)
        if old_to_new is None:
            return
        
        labels = list(old_to_new.values())

        new_column = self.view.dialog_new_column_name(name=f'{column} - Stratified')
        if new_column is None:
            return

        self.model.stratify(column=column, intervals=intervals, labels=labels, new_column=new_column)
        self.view.refresh_table()
    
    def convert_categorical_variable(self, df: pd.DataFrame, column: str):
        old_to_new = {value: value for value in df[column].unique()}
        
        old_to_new = self.view.dialog_convert(old_to_new=old_to_new)
        if old_to_new is None:
            return
        
        new_column = self.view.dialog_new_column_name(name=f'{column} - Converted')
        if new_column is None:
            return
        
        self.model.convert(column=column, old_to_new=old_to_new, new_column=new_column)
        self.view.refresh_table()


class ActionRenameColumn(Action):

    def action(self):
        columns = self.view.get_selected_columns()
        if len(columns) == 0:
            self.view.message_box_error(msg='Please select a column')
            return
        elif len(columns) > 1:
            self.view.message_box_error(msg='Please select only one column')
            return
        column = columns[0]
        new_name = self.view.dialog_rename_column(name=column)
        if new_name is None:
            return
        self.model.rename_column(column=column, new_name=new_name)
        self.view.refresh_table()
