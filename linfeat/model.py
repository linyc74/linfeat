import os
import datetime
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Tuple, Type
from .univariable import UnivariableStatistics
from .multivariable import MultivariableRegression
from .basic import determine_variable_type, BINARY, CATEGORICAL, CONTINUOUS


class DataPacket:

    df: pd.DataFrame
    column_to_type: Dict[str, str]
    column_to_parametric: Dict[str, bool]

    def __init__(self, df: pd.DataFrame, column_to_type: Dict[str, str], column_to_parametric: Dict[str, bool]):
        self.df = df
        self.column_to_type = column_to_type
        self.column_to_parametric = column_to_parametric


class Model:

    MAX_UNDO = 100

    dataframe: pd.DataFrame
    column_to_parametric: Dict[str, bool]
    active_file: Optional[str]

    undo_cache: List[pd.DataFrame]
    redo_cache: List[pd.DataFrame]

    def __init__(self):
        self.dataframe = pd.DataFrame()
        self.column_to_parametric = {}
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
            df = pd.read_excel(file)
        elif file.endswith('.csv'): 
            df = pd.read_csv(file)
        else:  # assume tab-separated file
            df = pd.read_csv(file, sep='\t')

        self.active_file = file

        matrix = df.to_numpy(dtype=object, copy=True)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = cast_to_appropriate_type(matrix[i, j])
        df = pd.DataFrame(matrix, index=df.index, columns=df.columns, dtype=object)

        self.__add_to_undo_cache()
        self.dataframe = df
        self.column_to_parametric = {column: False for column in df.columns}

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
        return DataPacket(df, column_to_type, column_to_parametric)

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
            if key in new.columns:
                new.loc[row, key] = cast_to_appropriate_type(value)
            else:
                raise ValueError(f'Column "{key}" not found in dataframe')

        self.__add_to_undo_cache()  # add to undo cache after successful update
        self.dataframe = new

    def append_row(self, attributes: Dict[str, str]):
        series = pd.Series({
            key: cast_to_appropriate_type(value)
            for key, value in attributes.items()
        })
        new = append(self.dataframe, series)

        self.__add_to_undo_cache()  # add to undo cache after successful append
        self.dataframe = new

    def update_cell(self, row: int, column: str, value: Any):
        new = self.dataframe.copy()
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

    def set_column_parametric(self, column: str, parametric: bool):
        self.column_to_parametric[column] = parametric

    def univariable_statistics(self, outdir: str, outcome: str, colors: List[str]):
        df = self.dataframe.copy()
        assert df[outcome].notna().all(), f'Outcome "{outcome}" has missing values'

        for c in df.columns:
            type_ = determine_variable_type(df[c])
            if type_ == BINARY:
                df[c] = df[c].astype(int)
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
    
    # str to float, if possible
    if isinstance(v, str):
        v = v.strip()  # remove preceding and trailing whitespace
        
        if v in ['', 'nan', 'NaN', 'N/A', 'n/a', 'na', 'NA', 'null', 'None', 'none']:
            v = np.nan  # np.nan is float
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
