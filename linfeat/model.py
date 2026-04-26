import os
import datetime
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Tuple, Type, Set
from .univariable import UnivariableStatistics
from .multivariable import MultivariableRegression
from .basic import determine_variable_type, BINARY, CATEGORICAL, CONTINUOUS


class DataPacket:

    df: pd.DataFrame
    column_to_type: Dict[str, str]
    column_to_parametric: Dict[str, bool]
    forced_categorical_columns: Set[str]

    def __init__(
            self,
            df: pd.DataFrame,
            column_to_type: Dict[str, str],
            column_to_parametric: Dict[str, bool],
            forced_categorical_columns: Set[str]):
        self.df = df
        self.column_to_type = column_to_type
        self.column_to_parametric = column_to_parametric
        self.forced_categorical_columns = forced_categorical_columns


class Model:

    MAX_UNDO = 100

    dataframe: pd.DataFrame
    forced_categorical_columns: Set[str]
    column_to_parametric: Dict[str, bool]
    active_file: Optional[str]

    undo_cache: List[pd.DataFrame]
    redo_cache: List[pd.DataFrame]

    def __init__(self):
        self.dataframe = pd.DataFrame()
        self.column_to_parametric = {}
        self.forced_categorical_columns = set()
        self.active_file = None
        self.undo_cache = []
        self.redo_cache = []

    def undo(self):
        if len(self.undo_cache) == 0:
            return
        self.redo_cache.append(self.dataframe)
        self.dataframe = self.undo_cache.pop()

    def redo(self):
        if len(self.redo_cache) == 0:
            return
        self.undo_cache.append(self.dataframe)
        self.dataframe = self.redo_cache.pop()

    def __add_to_undo_cache(self):
        self.undo_cache.append(self.dataframe.copy())
        if len(self.undo_cache) > self.MAX_UNDO:
            self.undo_cache.pop(0)
        self.redo_cache = []  # clear redo cache

    def open(self, file: str):
        if file.endswith('.xlsx'):
            df = pd.read_excel(file, keep_default_na=False, dtype=object)
        elif file.endswith('.csv'): 
            df = pd.read_csv(file, keep_default_na=False, dtype=object)
        else:  # assume tab-separated file
            df = pd.read_csv(file, sep='\t', keep_default_na=False, dtype=object)

        self.active_file = file

        matrix = df.to_numpy(dtype=object, copy=True)
        columns = df.columns.tolist()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                column = columns[j]
                value = matrix[i, j]
                if column in self.forced_categorical_columns:
                    matrix[i, j] = cast_to_categorical(value)  # categorical str
                else:
                    matrix[i, j] = cast_to_appropriate_type(value)
        df = pd.DataFrame(matrix, index=df.index, columns=df.columns, dtype=object)

        self.__add_to_undo_cache()
        self.dataframe = df
        self.column_to_parametric = {column: False for column in df.columns}
        self.forced_categorical_columns = set()

    def save(self, file: str):
        if file.endswith('.xlsx'):
            self.dataframe.to_excel(file, index=False)
        elif file.endswith('.csv'):
            self.dataframe.to_csv(file, index=False, encoding='utf-8-sig')
        else:  # assume tab-separated file
            self.dataframe.to_csv(file, index=False, sep='\t', encoding='utf-8-sig')

        self.active_file = file

    def get_data_packet(self) -> DataPacket:
        df = self.dataframe.copy()
        column_to_type = {c: determine_variable_type(df[c]) for c in df.columns}
        column_to_parametric = self.column_to_parametric.copy()
        forced_categorical_columns = self.forced_categorical_columns.copy()
        return DataPacket(df, column_to_type, column_to_parametric, forced_categorical_columns)

    def sort_dataframe(
            self,
            by: str,
            ascending: bool):
        new = self.dataframe.copy()

        # str cannot be compared with float/int, so we need to convert all to str in that case
        dtypes = set()
        for value in new[by]:
            dtypes.add(type(value))
        if str in dtypes and (float in dtypes or int in dtypes):
            new['sorting'] = new[by].astype(str)
            by = 'sorting'

        new = new.sort_values(
            by=by,
            ascending=ascending,
            kind='mergesort'  # deterministic, keep the original order when tied
        ).reset_index(
            drop=True
        )

        if by == 'sorting':
            new.drop(columns=['sorting'], inplace=True)

        self.__add_to_undo_cache()  # add to undo cache after successful sort
        self.dataframe = new

    def drop(
            self,
            rows: Optional[List[int]] = None,
            columns: Optional[List[str]] = None):
        new = self.dataframe.drop(
            index=rows,
            columns=columns
        ).reset_index(
            drop=True
        )

        if new.shape[0] == 0:
            raise ValueError('Cannot drop all rows.')

        if new.shape[1] == 0:
            raise ValueError('Cannot drop all columns.')
            
        self.__add_to_undo_cache()  # add to undo cache after successful drop
        self.dataframe = new

        if columns is not None:
            for column in columns:
                self.column_to_parametric.pop(column)

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
        new = self.dataframe.copy()
        for key, value in attributes.items():
            if key not in new.columns:
                raise ValueError(f'Column "{key}" not found in dataframe')
            if key in self.forced_categorical_columns:
                new.loc[row, key] = cast_to_categorical(value)  # categorical str
            else:
                new.loc[row, key] = cast_to_appropriate_type(value)

        self.__add_to_undo_cache()  # add to undo cache after successful update
        self.dataframe = new

    def append_row(self, attributes: Dict[str, str]):
        series = []
        for key, value in attributes.items():
            if key in self.forced_categorical_columns:
                series.append(cast_to_categorical(value))  # categorical str
            else:
                series.append(cast_to_appropriate_type(value))
        series = pd.Series(data=series, dtype=object)
        new = append(self.dataframe, series)

        self.__add_to_undo_cache()  # add to undo cache after successful append
        self.dataframe = new

    def update_cell(self, row: int, column: str, value: Any):
        new = self.dataframe.copy()
        if column in self.forced_categorical_columns:
            new.loc[row, column] = cast_to_categorical(value)
        else:
            new.loc[row, column] = cast_to_appropriate_type(value)

        self.__add_to_undo_cache()  # add to undo cache after successful update
        self.dataframe = new

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
        for c, parametric in column_to_parametric.items():
            if c in self.forced_categorical_columns:
                # this is a forced categorical variable
                # do not change its parametric property which might be already set by the user
                continue
            elif determine_variable_type(self.dataframe[c]) == CONTINUOUS:
                # only continous variable can be defined as parametric or nonparametric
                self.column_to_parametric[c] = parametric
            else:
                # categorical or binary variable can only be nonparametric
                # do not change its parametric property
                continue

    def stratify(
            self,
            column: str,
            intervals: List[Tuple[float, float]],
            labels: List[str],
            new_column: str):

        assert len(labels) == len(intervals), f'Number of labels must match number of intervals. labels: {labels}; intervals: {intervals}.'
        assert new_column not in self.dataframe.columns, f'Column "{new_column}" already exists.'

        df = self.dataframe.copy()
        
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

        self.__add_to_undo_cache()  # add to undo cache after successful stratify
        self.dataframe = df
        self.column_to_parametric[new_column] = False

    def convert(
            self,
            column: str,
            old_to_new: Dict[Any, str],
            new_column: str):
        assert new_column not in self.dataframe.columns, f'Column "{new_column}" already exists.'

        df = self.dataframe.copy()

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

        self.__add_to_undo_cache()  # add to undo cache after successful convert
        self.dataframe = df
        self.column_to_parametric[new_column] = False

    def add_column(self, column: str):
        assert column not in self.dataframe.columns, f'Column "{column}" already exists'
        df = self.dataframe.copy()
        df[column] = pd.Series(data=np.nan, dtype=object)
        self.__add_to_undo_cache()  # add to undo cache after successful add
        self.dataframe = df
        self.column_to_parametric[column] = False

    def rename_column(self, column: str, new_name: str):
        if new_name == column:
            return

        df = self.dataframe.copy()
        assert new_name not in df.columns, f'Column "{new_name}" already exists'
        df.rename(columns={column: new_name}, inplace=True)

        self.__add_to_undo_cache()  # add to undo cache after successful rename
        self.dataframe = df

        self.column_to_parametric[new_name] = self.column_to_parametric[column]
        self.column_to_parametric.pop(column)

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
        df = self.dataframe.copy()
        
        df_changed = False
        for column in columns:
            if column in self.forced_categorical_columns:
                continue
            df_changed = True
            series = [cast_to_categorical(v) for v in df[column]]
            df[column] = pd.Series(data=series, dtype=object)  # always ensure object dtype

        if not df_changed:
            return

        self.__add_to_undo_cache()  # add to undo cache after successful force categorical
        self.dataframe = df
        for column in columns:
            self.forced_categorical_columns.add(column)

    def unforce_categorical(self, columns: List[str]):
        df = self.dataframe.copy()

        df_changed = False
        for column in columns:
            if column not in self.forced_categorical_columns:
                continue
            df_changed = True
            series = [cast_to_appropriate_type(v) for v in df[column]]
            df[column] = pd.Series(data=series, dtype=object)  # always ensure object dtype

        if not df_changed:
            return

        self.__add_to_undo_cache()  # add to undo cache after successful unforce categorical
        self.dataframe = df
        for column in columns:
            self.forced_categorical_columns.remove(column)

    def fill_missing_values(self, binary: str, continuous: str, categorical: str):
        df = self.dataframe.copy()

        assert binary in ['0', '1'], f'Binary must be 0 or 1. Got "{binary}".'
        assert (continuous.lower() in ['mean', 'median']) or continuous.isdigit(), f'Continuous must be mean, median, or a numeric value. Got "{continuous}".'

        for column in df.columns:
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

        self.__add_to_undo_cache()  # add to undo cache after successful fill missing values
        self.dataframe = df


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
