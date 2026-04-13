import os
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Tuple, Type


class Model:

    MAX_UNDO = 100

    dataframe: pd.DataFrame  # this is the main clinical data table
    active_file: Optional[str]

    undo_cache: List[pd.DataFrame]
    redo_cache: List[pd.DataFrame]

    def __init__(self):
        self.dataframe = pd.DataFrame()
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
            new = pd.read_excel(file)
        elif file.endswith('.csv'): 
            new = pd.read_csv(file)
        else:  # assume tab-separated file
            new = pd.read_csv(file, sep='\t')

        self.active_file = file

        for column in new.columns:
            new[column] = new[column].astype(object)  # object, to allow mixed types

        for idx in new.index:
            for column in new.columns:
                value = new.loc[idx, column]
                new.loc[idx, column] = cast_to_appropriate_type(value)

        self.__add_to_undo_cache()
        self.dataframe = new

    def save(self, file: str):
        if file.endswith('.xlsx'):
            self.dataframe.to_excel(file, index=False)
        elif file.endswith('.csv'):
            self.dataframe.to_csv(file, index=False, encoding='utf-8-sig')
        else:  # assume tab-separated file
            self.dataframe.to_csv(file, index=False, sep='\t', encoding='utf-8-sig')

        self.active_file = file

    def get_data_packet(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        df = self.dataframe.copy()
        column_to_type = {}
        for column in df.columns:
            column_to_type[column] = determine_variable_type(df[column])
        return df, column_to_type

    def sort_dataframe(
            self,
            by: str,
            ascending: bool):
        new = self.dataframe.sort_values(
            by=by,
            ascending=ascending,
            kind='mergesort'  # deterministic, keep the original order when tied
        ).reset_index(
            drop=True
        )
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
        self.__add_to_undo_cache()  # add to undo cache after successful drop
        self.dataframe = new

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


#


from typing import Iterable


BINARY = 'binary'
CONTINUOUS = 'continuous'
CATEGORICAL = 'categorical'


def determine_variable_type(series: Iterable[Any]) -> str:
    __series = [v for v in series if not pd.isna(v)]

    if len(__series) == 0:
        raise ValueError(f'"{series}" has no valid values to determine variable type')

    type_series = []

    for v in __series:
        if isinstance(v, int) or isinstance(v, float):
            if v == 0 or v == 1:
                type_series.append(1)  # 1 means binary
            else:
                type_series.append(2)  # 2 means continuous
        elif isinstance(v, str):
            type_series.append(3)  # 3 means categorical
        else:
            type_series.append(3)  # others (e.g. bool) default to 3, categorical

    if max(type_series) == 1:
        return BINARY
    elif max(type_series) == 2:
        return CONTINUOUS
    else:
        return CATEGORICAL
