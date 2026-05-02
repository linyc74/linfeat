import os
import datetime
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Tuple, Type, Set
from .normality import Normality
from .univariable import UnivariableStatistics
from .multivariable import MultivariableRegression
from .basic import determine_variable_type, BINARY, CATEGORICAL, CONTINUOUS


class DataPacket:

    df: pd.DataFrame
    column_to_type: Dict[str, str]
    column_to_parametric: Dict[str, bool]
    forced_categorical_columns: Set[str]
    column_to_summary: Dict[str, str]

    def __init__(
            self,
            df: pd.DataFrame,
            column_to_type: Dict[str, str],
            column_to_parametric: Dict[str, bool],
            forced_categorical_columns: Set[str],
            column_to_summary: Dict[str, str]):
        self.df = df
        self.column_to_type = column_to_type
        self.column_to_parametric = column_to_parametric
        self.forced_categorical_columns = forced_categorical_columns
        self.column_to_summary = column_to_summary


class Model:

    MAX_UNDO = 100

    # these 3 are cached simultaneously for undo/redo
    dataframe: pd.DataFrame
    forced_categorical_columns: Set[str]
    column_to_parametric: Dict[str, bool]

    undo_cache: List[Tuple[pd.DataFrame, Set[str], Dict[str, bool]]]
    redo_cache: List[Tuple[pd.DataFrame, Set[str], Dict[str, bool]]]

    # not cached, only the current state
    saved_dataframe_id: Optional[int]

    def __init__(self):
        self.dataframe = pd.DataFrame()
        self.forced_categorical_columns = set()
        self.column_to_parametric = {}
        self.saved_dataframe_id = id(self.dataframe)  # initial state is set to saved
        self.undo_cache = []
        self.redo_cache = []

    def undo(self):
        if len(self.undo_cache) == 0:
            return
        self.redo_cache.append((self.dataframe, self.forced_categorical_columns, self.column_to_parametric))
        self.dataframe, self.forced_categorical_columns, self.column_to_parametric = self.undo_cache.pop()

    def redo(self):
        if len(self.redo_cache) == 0:
            return
        self.undo_cache.append((self.dataframe, self.forced_categorical_columns, self.column_to_parametric))
        self.dataframe, self.forced_categorical_columns, self.column_to_parametric = self.redo_cache.pop()

    def _copy_state(self) -> Tuple[pd.DataFrame, Set[str], Dict[str, bool]]:
        return (self.dataframe.copy(), self.forced_categorical_columns.copy(), self.column_to_parametric.copy())

    def _move_state_forward(
            self,
            dataframe: pd.DataFrame,
            forced_categorical_columns: Set[str],
            column_to_parametric: Dict[str, bool]):
        # add current state to undo cache
        self.undo_cache.append((self.dataframe, self.forced_categorical_columns, self.column_to_parametric))
        if len(self.undo_cache) > self.MAX_UNDO:
            self.undo_cache.pop(0)
        
        # clear redo cache
        self.redo_cache = []

        # update current state
        self.dataframe = dataframe
        self.forced_categorical_columns = forced_categorical_columns
        self.column_to_parametric = column_to_parametric

    def open(self, file: str):
        if file.endswith('.xlsx'):
            df = pd.read_excel(file, keep_default_na=False, dtype=object)
        elif file.endswith('.csv'): 
            df = pd.read_csv(file, keep_default_na=False, dtype=object)
        else:  # assume tab-separated file
            df = pd.read_csv(file, sep='\t', keep_default_na=False, dtype=object)

        matrix = df.to_numpy(dtype=object, copy=True)
        columns = df.columns.tolist()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                column = columns[j]
                value = matrix[i, j]
                matrix[i, j] = cast_to_appropriate_type(value)
        df = pd.DataFrame(matrix, index=df.index, columns=df.columns, dtype=object)

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=set(),
            column_to_parametric={column: False for column in df.columns}
        )

    def save(self, file: str):
        if file.endswith('.xlsx'):
            self.dataframe.to_excel(file, index=False)
        elif file.endswith('.csv'):
            self.dataframe.to_csv(file, index=False, encoding='utf-8-sig')
        else:  # assume tab-separated file
            self.dataframe.to_csv(file, index=False, sep='\t', encoding='utf-8-sig')        
        self.saved_dataframe_id = id(self.dataframe)  # update saved dataframe id after successful save

    def get_data_packet(self) -> DataPacket:
        df, forced_categorical_columns, column_to_parametric = self._copy_state()
        column_to_type = {c: determine_variable_type(df[c]) for c in df.columns}
        column_to_summary = generate_column_to_summary(df)
        return DataPacket(df, column_to_type, column_to_parametric, forced_categorical_columns, column_to_summary)

    def sort_dataframe(
            self,
            by: str,
            ascending: bool):
        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        # str cannot be compared with float/int, so we need to convert all to str in that case
        dtypes = set()
        for value in df[by]:
            dtypes.add(type(value))
        if str in dtypes and (float in dtypes or int in dtypes):
            df['sorting'] = df[by].astype(str)
            by = 'sorting'

        df = df.sort_values(
            by=by,
            ascending=ascending,
            kind='mergesort'  # deterministic, keep the original order when tied
        ).reset_index(
            drop=True
        )

        if by == 'sorting':
            df.drop(columns=['sorting'], inplace=True)

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def drop(
            self,
            rows: Optional[List[int]] = None,
            columns: Optional[List[str]] = None):
        df, forced_categorical_columns, column_to_parametric = self._copy_state()
        
        df = df.drop(
            index=rows,
            columns=columns
        ).reset_index(
            drop=True
        )

        if df.shape[0] == 0:
            raise ValueError('Cannot drop all rows.')

        if df.shape[1] == 0:
            raise ValueError('Cannot drop all columns.')

        if columns is not None:
            for c in columns:
                if c in forced_categorical_columns:
                    forced_categorical_columns.remove(c)
                if c in column_to_parametric:
                    column_to_parametric.pop(c)
            
        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def get_row(self, row: int) -> Dict[str, Any]:
        ret = self.dataframe.loc[row].to_dict()

        for key, val in ret.items():
            if pd.isna(val):
                ret[key] = ''  # NaN to ''

        return ret

    def get_value(self, row: int, column: str) -> Any:
        val = self.dataframe.loc[row, column]
        return '' if pd.isna(val) else val

    def update_row(self, row: int, attributes: Dict[str, str]):
        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        for key, value in attributes.items():
            if key not in df.columns:
                raise ValueError(f'Column "{key}" not found in dataframe')
            if key in forced_categorical_columns:
                df.loc[row, key] = cast_to_categorical(value)  # categorical str
            else:
                df.loc[row, key] = cast_to_appropriate_type(value)

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def append_row(self, attributes: Dict[str, str]):
        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        row = {}
        for key, value in attributes.items():
            if key in forced_categorical_columns:
                row[key] = cast_to_categorical(value)  # categorical str
            else:
                row[key] = cast_to_appropriate_type(value)
        row = pd.Series(data=row, dtype=object)
        df = append(df, row)

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def update_cell(self, row: int, column: str, value: Any):
        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        if column in self.forced_categorical_columns:
            df.loc[row, column] = cast_to_categorical(value)
        else:
            df.loc[row, column] = cast_to_appropriate_type(value)

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def find(self, text: str, start: Optional[Tuple[int, str]]) -> Optional[Tuple[int, str]]:
        if start is None:
            start_irow = 0
            start_icol = 0
        else:
            start_irow = start[0]
            start_icol = self.dataframe.columns.to_list().index(start[1])

        nrows, ncols = self.dataframe.shape

        for r in range(nrows):
            for c in range(ncols):
                if r <= start_irow and c <= start_icol:
                    continue
                if text.lower() in str(self.dataframe.iloc[r, c]).lower():
                    return r, self.dataframe.columns[c]

    def set_parametric_properties(self, column_to_parametric: Dict[str, bool]):
        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        for c, parametric in column_to_parametric.items():
            if c in forced_categorical_columns:
                # this is a forced categorical variable
                # do not change its parametric property which might be already set by the user
                continue
            elif determine_variable_type(df[c]) == CONTINUOUS:
                # only continous variable can be defined as parametric or nonparametric
                column_to_parametric[c] = parametric
        
        if column_to_parametric == self.column_to_parametric:
            # no change, do not need to add to undo cache
            return
        
        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def stratify(
            self,
            column: str,
            intervals: List[Tuple[float, float]],
            labels: List[str],
            new_column: str):

        assert len(labels) == len(intervals), f'Number of labels must match number of intervals. labels: {labels}; intervals: {intervals}.'
        assert new_column not in self.dataframe.columns, f'Column "{new_column}" already exists.'

        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        df[new_column] = pd.Series(data=np.nan, dtype=object)  # instantiate with object dtype
        for interval, label in zip(intervals, labels):
            a, b = interval
            within_interval = (a <= df[column]) & (df[column] < b)
            df.loc[within_interval, new_column] = cast_to_appropriate_type(label)
        
        # fill in the value that equals the upper bound
        upper_bound = intervals[-1][1]
        df.loc[df[column] == upper_bound, new_column] = cast_to_appropriate_type(labels[-1])

        # move the new column to the right of the original column
        columns = df.columns.tolist()
        pos = columns.index(column) + 1
        reordered = columns[:pos] + [new_column] + columns[pos:-1]
        df = df[reordered]

        column_to_parametric[new_column] = False

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def convert(
            self,
            column: str,
            old_to_new: Dict[Any, str],
            new_column: str):
        assert new_column not in self.dataframe.columns, f'Column "{new_column}" already exists.'

        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        new_values = []
        for old in df[column]:
            new = old_to_new.get(old, np.nan)
            new_values.append(cast_to_appropriate_type(new))

        df[new_column] = pd.Series(data=new_values, dtype=object)
        
        # move the new column to the right of the original column
        columns = df.columns.tolist()
        pos = columns.index(column) + 1
        reordered = columns[:pos] + [new_column] + columns[pos:-1]
        df = df[reordered]

        column_to_parametric[new_column] = False

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def add_column(self, column: str):
        assert column not in self.dataframe.columns, f'Column "{column}" already exists'
        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        df[column] = pd.Series(data=np.nan, dtype=object)
        column_to_parametric[column] = False

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def rename_column(self, column: str, new_name: str):
        if new_name == column:
            return

        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        assert new_name not in df.columns, f'Column "{new_name}" already exists'
        df.rename(columns={column: new_name}, inplace=True)

        if column in forced_categorical_columns:
            forced_categorical_columns.add(new_name)
            forced_categorical_columns.remove(column)
        if column in column_to_parametric:
            column_to_parametric[new_name] = column_to_parametric[column]
            column_to_parametric.pop(column)

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def univariable_statistics(self, outdir: str, outcome: str, colors: List[str]):
        df = self.dataframe.copy()
        assert df[outcome].notna().all(), f'Outcome "{outcome}" has missing values'

        for c in df.columns:
            type_ = determine_variable_type(df[c])
            if type_ == BINARY:
                if df[c].isna().any():
                    df[c] = df[c].astype(float)  # missing values (np.nan) cannot be converted to int
                else:
                    df[c] = df[c].astype(int)  # no missing, convert all to int
            elif type_ == CATEGORICAL:
                df[c] = df[c].astype(str)
            elif type_ == CONTINUOUS:
                df[c] = df[c].astype(float)

        variables = [c for c in df.columns if c != outcome]
        parametric_variables = [c for c in variables if self.column_to_parametric[c]]
        UnivariableStatistics().main(
            df=df,
            variables=variables,
            outcome=outcome,
            outdir=outdir,
            parametric_outcome=self.column_to_parametric[outcome],
            parametric_variables=parametric_variables,
            colors=colors,
        )

    def multivariable_regression(self, outdir: str, outcome: str):
        df = self.dataframe.copy().astype(float)
        MultivariableRegression().main(
            df=df,
            variables=[c for c in df.columns if c != outcome],
            outcome=outcome,
            outdir=outdir,
        )
    
    def force_categorical(self, columns: List[str]):
        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        # only force categorical for columns that are not already forced categorical
        columns = [c for c in columns if c not in forced_categorical_columns]
        if len(columns) == 0:
            return
        
        for column in columns:
            series = [cast_to_categorical(v) for v in df[column]]
            df[column] = pd.Series(data=series, dtype=object)  # always ensure object dtype
            forced_categorical_columns.add(column)

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def unforce_categorical(self, columns: List[str]):
        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        # only unforce categorical for columns that are already forced categorical
        columns = [c for c in columns if c in forced_categorical_columns]
        if len(columns) == 0:
            return

        for column in columns:
            series = [cast_to_appropriate_type(v) for v in df[column]]
            df[column] = pd.Series(data=series, dtype=object)  # always ensure object dtype
            forced_categorical_columns.remove(column)        

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)

    def fill_missing_values(self, columns: List[str], binary: str, continuous: str, categorical: str):
        df, forced_categorical_columns, column_to_parametric = self._copy_state()

        assert binary in ['0', '1'], f'Binary must be 0 or 1. Got "{binary}".'
        assert (continuous.lower() in ['mean', 'median']) or continuous.isdigit(), f'Continuous must be mean, median, or a numeric value. Got "{continuous}".'

        for column in columns:
            if not df[column].isna().any():
                continue

            if column in self.forced_categorical_columns:
                type_ = CATEGORICAL
            else:
                type_ = determine_variable_type(df[column])
            
            if type_ == BINARY:
                df.loc[df[column].isna(), column] = int(binary)  # always int

            elif type_ == CONTINUOUS:
                if continuous.lower() == 'mean':
                    v = df[column].mean()  # float
                elif continuous.lower() == 'median':
                    v = df[column].median()  # float
                else:
                    v = float(continuous)
                df.loc[df[column].isna(), column] = cast_to_appropriate_type(v)  # can be converted to int

            elif type_ == CATEGORICAL:
                df.loc[df[column].isna(), column] = categorical  # remains str, do not convert to int/float

        self._move_state_forward(
            dataframe=df,
            forced_categorical_columns=forced_categorical_columns,
            column_to_parametric=column_to_parametric)
    
    def normality_test(
            self,
            shapiro_p: str,
            kolmogorov_p: str,
            skewness: str,
            excess_kurtosis: str,
            outdir: str):

        df = self.dataframe.copy()

        shapiro_p = float(shapiro_p)
        kolmogorov_p = float(kolmogorov_p)
        skewness = float(skewness)
        excess_kurtosis = float(excess_kurtosis)

        continuous_variables = []
        for c in df.columns:
            if determine_variable_type(df[c]) == CONTINUOUS:
                continuous_variables.append(c)
                df[c] = df[c].astype(float)
        
        if len(continuous_variables) == 0:
            raise ValueError('No continuous variables found')

        stats_df = Normality().main(
            df=df,
            variables=continuous_variables,
            shapiro_p_threshold=shapiro_p,
            kolmogorov_p_threshold=kolmogorov_p,
            skewness_threshold=skewness,
            excess_kurtosis_threshold=excess_kurtosis,
            outdir=outdir,
        )

        variables_passed = stats_df[stats_df['Pass Normality Test']]['Variable'].tolist()
        variables_failed = stats_df[~stats_df['Pass Normality Test']]['Variable'].tolist()

        for variable in variables_passed:
            self.column_to_parametric[variable] = True
        for variable in variables_failed:
            self.column_to_parametric[variable] = False

    def is_current_dataframe_saved(self) -> bool:
        return id(self.dataframe) == self.saved_dataframe_id


def append(
        df: pd.DataFrame,
        s: Union[dict, pd.Series]) -> pd.DataFrame:

    if type(s) is dict:
        s = pd.Series(s)

    if df.empty:
        return pd.DataFrame([s])  # no need to concat, just return the Series as a DataFrame

    return pd.concat([df, pd.DataFrame([s])], ignore_index=True)


def cast_to_appropriate_type(value: Any) -> Any:
    v = value

    if pd.isna(v):
        return np.nan
    
    # str to float, if possible
    if isinstance(v, str):
        v = v.strip()  # remove preceding and trailing whitespace
        
        if v == '':
            v = np.nan  # np.nan is float
        elif v.lower() == 'nan':
            v = 'nan'  # avoid converting to np.nan, keep as str because the user may want to show it as is
        else:
            try:
                v = float(v)
            except ValueError:
                pass

    # float to int, if possible
    if isinstance(v, float):
        if v.is_integer():
            v = int(v)

    return v


def cast_to_categorical(value: Any) -> Union[str, float]:
    v = value

    if pd.isna(v):
        return np.nan

    if isinstance(v, str):
        v = v.strip()
        if v == '':
            return np.nan

    return str(v)


def generate_column_to_summary(df: pd.DataFrame) -> Dict[str, str]:
    ret = {}
    for column in df.columns:
        type_ = determine_variable_type(df[column])
        summary = f'{column}\n\n{type_.capitalize()} variable\n\n'
        if type_ in [BINARY, CATEGORICAL]:
            for value, count in df[column].value_counts().to_dict().items():
                summary += f'{value}: {count}\n'
            summary += f'\nN: {df[column].count()}'
        elif type_ == CONTINUOUS:
            summary += f'Mean ± SD: {df[column].mean():.3g} ± {df[column].std():.3g}\n'
            summary += f'Min: {df[column].min():.3g}\n'
            summary += f'Max: {df[column].max():.3g}\n'
            summary += f'N: {df[column].count()}'
        else:
            raise ValueError(f'Unknown variable type "{type_}" of the column "{column}"')
        ret[column] = summary.strip()
    return ret
