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
        self.__connect_drag_drop_actions()

    def __init_actions(self):
        self.action_open = ActionOpen(self)
        self.action_open_dropped_file = ActionOpenDroppedFile(self)
        self.action_save_as = ActionSaveAs(self)
        self.action_add_new_row = ActionAddNewRow(self)
        self.action_edit_row = ActionEditRow(self)
        self.action_edit_cell = ActionEditCell(self)
        self.action_undo = ActionUndo(self)
        self.action_redo = ActionRedo(self)
        self.action_find = ActionFind(self)
        self.action_add_new_column = ActionAddNewColumn(self)
        self.action_rename_column = ActionRenameColumn(self)
        self.action_sort_ascending = ActionSortAscending(self)
        self.action_sort_descending = ActionSortDescending(self)
        self.action_delete_rows = ActionDeleteRows(self)
        self.action_delete_columns = ActionDeleteColumns(self)
        self.action_control_s = ActionSaveAs(self)
        self.action_control_f = ActionFind(self)
        self.action_control_z = ActionUndo(self)
        self.action_control_y = ActionRedo(self)
        self.action_set_parametric_variables = ActionSetParametricVariables(self)
        self.action_univariable_statistics = ActionUnivariableStatistics(self)
        self.action_multivariable_regression = ActionMultivariableRegression(self)
        self.action_stratify_convert = ActionStratifyConvert(self)
        self.action_force_categorical = ActionForceCategorical(self)
        self.action_unforce_categorical = ActionUnforceCategorical(self)
        self.action_fill_missing_values = ActionFillMissingValues(self)
        self.action_normality_test = ActionNormalityTest(self)

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

    def __connect_drag_drop_actions(self):
        self.view.table.file_dropped.connect(self.action_open_dropped_file)


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
            self.view.error_message(msg=repr(e))
        
    def action(self) -> None:
        raise NotImplementedError('Action method not implemented')


class ActionOpen(Action):

    def action(self):
        file = self.view.open_file_dialog()
        if file == '':
            return
        self.model.open(file=file)
        self.view.refresh_table()


class ActionOpenDroppedFile(Action):

    def __call__(self, file: str):  # override the parent class, to take the `file` argument
        try:
            self.model.open(file=file)
            self.view.refresh_table()
        except Exception as e:
            self.view.error_message(msg=repr(e))


class ActionSaveAs(Action):

    def action(self):
        file = self.view.save_as_file_dialog(filename='New Table.xlsx')
        if file == '':
            return
        self.model.save(file=file)


class ActionFind(Action):

    def action(self):
        text = self.view.find_dialog()
        if text is None:
            return

        selected_cells = self.view.get_selected_cells()
        start = None if len(selected_cells) == 0 else selected_cells[0]

        found_cell = self.model.find(text=text, start=start)

        if found_cell is None:
            self.view.info_message(msg='Couldn\'t find what you were looking for')
            return

        index, column = found_cell
        self.view.select_cell(index=index, column=column)


class ActionSort(Action):

    ASCENDING: bool

    def action(self):
        columns = self.view.get_selected_columns()
        if len(columns) == 0:
            self.view.error_message(msg='Please select a column')
        elif len(columns) == 1:
            self.model.sort_dataframe(by=columns[0], ascending=self.ASCENDING)
            self.view.refresh_table()
        else:
            self.view.error_message(msg='Please select only one column')


class ActionSortAscending(ActionSort):

    ASCENDING = True


class ActionSortDescending(ActionSort):

    ASCENDING = False


class ActionDeleteRows(Action):

    def action(self):
        rows = self.view.get_selected_rows()
        if len(rows) == 0:
            return
        if self.view.yes_no_message(msg='Are you sure you want to delete the selected rows?'):
            self.model.drop(rows=rows)
            self.view.refresh_table()


class ActionDeleteColumns(Action):

    def action(self):
        columns = self.view.get_selected_columns()
        if len(columns) == 0:
            return
        if self.view.yes_no_message(msg='Are you sure you want to delete the selected columns?'):
            self.model.drop(columns=columns)
            self.view.refresh_table()


class ActionAddNewRow(Action):

    def action(self):
        attributes = self.view.edit_row_dialog(attributes=None)
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
            self.view.error_message(msg='Please select a row')
            return
        elif len(rows) > 1:
            self.view.error_message(msg='Please select only one row')
            return

        row = rows[0]  # only one row is selected

        attributes = self.model.get_row(row=row)

        attributes = self.view.edit_row_dialog(attributes=attributes)
        if attributes is None:
            return
        self.model.update_row(row=row, attributes=attributes)

        self.view.refresh_table()


class ActionEditCell(Action):

    def action(self):
        cells = self.view.get_selected_cells()

        if len(cells) == 0:
            self.view.error_message('Please select a cell')
            return
        elif len(cells) > 1:
            self.view.error_message('Please select only one cell')
            return

        row, column = cells[0]
        value = self.model.get_value(row=row, column=column)

        new_value = self.view.edit_cell_dialog(value=value)

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
        variable_to_parametric = self.view.set_parametric_variables_dialog()
        if variable_to_parametric is None:
            return
        self.model.set_parametric_properties(column_to_parametric=variable_to_parametric)


class ActionUnivariableStatistics(Action):

    def action(self):
        outdir = self.view.open_directory_dialog(caption='Select Output Directory')
        if outdir == '':
            return
        outcome = self.view.select_outcome_dialog()
        if outcome is None:
            return
        colors = self.view.color_dialog()
        if colors is None:
            return
        self.model.univariable_statistics(outdir=outdir, outcome=outcome, colors=colors)
        self.view.info_message(msg='Univariable statistics completed')


class ActionMultivariableRegression(Action):

    def action(self):
        outdir = self.view.open_directory_dialog(caption='Select Output Directory')
        if outdir == '':
            return
        outcome = self.view.select_outcome_dialog()
        if outcome is None:
            return
        self.model.multivariable_regression(outdir=outdir, outcome=outcome)
        self.view.info_message(msg='Multivariable regression completed')


class ActionStratifyConvert(Action):

    def action(self):
        columns = self.view.get_selected_columns()
        if len(columns) == 0:
            self.view.error_message(msg='Please select a column')
            return
        elif len(columns) > 1:
            self.view.error_message(msg='Please select only one column')
            return
        column = columns[0]

        packet = self.model.get_data_packet()
        df = packet.df
        if packet.column_to_type[column] == CONTINUOUS:
            self.stratify_continous_variable(df=df, column=column)
        else:
            self.convert_categorical_variable(df=df, column=column)

    def stratify_continous_variable(self, df: pd.DataFrame, column: str):
        cutoffs = self.view.stratify_dialog(minimum=df[column].min(), maximum=df[column].max())
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

        old_to_new = self.view.convert_dialog(old_to_new=old_to_new)
        if old_to_new is None:
            return
        
        labels = list(old_to_new.values())

        new_column = self.view.new_column_name_dialog(name=f'{column} - Stratified')
        if new_column is None:
            return

        self.model.stratify(column=column, intervals=intervals, labels=labels, new_column=new_column)
        self.view.refresh_table()
    
    def convert_categorical_variable(self, df: pd.DataFrame, column: str):
        old_to_new = {value: value for value in df[column].unique()}
        
        old_to_new = self.view.convert_dialog(old_to_new=old_to_new)
        if old_to_new is None:
            return
        
        new_column = self.view.new_column_name_dialog(name=f'{column} - Converted')
        if new_column is None:
            return
        
        self.model.convert(column=column, old_to_new=old_to_new, new_column=new_column)
        self.view.refresh_table()


class ActionAddNewColumn(Action):

    def action(self):
        column = self.view.new_column_name_dialog(name='New Column')
        if column is None:
            return
        self.model.add_column(column=column)
        self.view.refresh_table()


class ActionRenameColumn(Action):

    def action(self):
        columns = self.view.get_selected_columns()
        if len(columns) == 0:
            self.view.error_message(msg='Please select a column')
            return
        elif len(columns) > 1:
            self.view.error_message(msg='Please select only one column')
            return
        column = columns[0]
        new_name = self.view.rename_column_dialog(name=column)
        if new_name is None:
            return
        self.model.rename_column(column=column, new_name=new_name)
        self.view.refresh_table()


class ActionForceCategorical(Action):

    def action(self):
        columns = self.view.get_selected_columns()
        if len(columns) == 0:
            self.view.error_message(msg='Please select columns')
            return
        self.model.force_categorical(columns=columns)
        self.view.refresh_table()


class ActionUnforceCategorical(Action):

    def action(self):
        columns = self.view.get_selected_columns()
        if len(columns) == 0:
            self.view.error_message(msg='Please select columns')
            return
        self.model.unforce_categorical(columns=columns)
        self.view.refresh_table()


class ActionFillMissingValues(Action):

    def action(self):
        defaults = self.view.fill_missing_values_dialog()
        if defaults is None:
            return
        binary, continuous, categorical = defaults
        self.model.fill_missing_values(
            binary=binary,
            continuous=continuous,
            categorical=categorical
        )
        self.view.refresh_table()


class ActionNormalityTest(Action):

    def action(self):
        thresholds = self.view.normality_test_dialog()
        if thresholds is None:
            return
        shapiro_p, kolmogorov_p, skewness, excess_kurtosis = thresholds

        outdir = self.view.open_directory_dialog(caption='Select Output Directory')
        if outdir == '':
            return
        
        self.model.normality_test(
            shapiro_p=shapiro_p,
            kolmogorov_p=kolmogorov_p,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            outdir=outdir,
        )
        self.view.info_message(msg='Normality test completed\nSet parametric variables accordingly')
