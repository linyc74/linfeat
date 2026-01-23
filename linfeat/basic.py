import matplotlib
import numpy as np
import pandas as pd
from typing import List, Iterable, Any


class Parameters:

    max_iter = 5000
    tol = 1e-4
    random_state = 42
    cv_folds = 5
    class_weight = 'balanced'

    l1_show_top_m_features = 5
    l1_alpha_min = 10**-2
    l1_alpha_max = 10**2
    l1_alpha_grid_steps = 100
    l1_c_min = 10**-2  # C = 1/alpha
    l1_c_max = 10**2
    l1_c_grid_steps = 100

    stepwise_n_features = 5  
    stepwise_alpha = 1.0
    stepwise_c = 1.0  # C = 1/alpha

    fig_width = 14 / 2.54
    fig_height = 7 / 2.54
    font_size = 7

    outdir = './outdir'


class PrepareData:

    df: pd.DataFrame
    features: List[str]
    outcome: str

    def main(self, df: pd.DataFrame, features: List[str], outcome: str) -> pd.DataFrame:
        self.df = df.copy()
        self.features = features
        self.outcome = outcome

        self.df = self.df[[self.outcome] + self.features]
        self.drop_samples_without_outcome()
        has_missing_values = self.check_missing_values()
        if has_missing_values:
            self.fill_missing_values()

        return self.df

    def drop_samples_without_outcome(self):
        indexes_without_outcome = self.df[self.df[self.outcome].isna()].index
        s = 's' if len(indexes_without_outcome) > 1 else ''
        print(f'Dropping {len(indexes_without_outcome)} sample{s} without outcome: {", ".join(indexes_without_outcome)}\n')
        self.df = self.df[self.df[self.outcome].notna()]

    def check_missing_values(self) -> bool:
        d = {}
        for column in self.df.columns:
            nan_count = self.df[column].isna().sum()
            if nan_count > 0:
                d[column] = nan_count

        if len(d) == 0:
            print('No missing values found.\n')
            return False

        sort_by_nan_count = sorted(d.items(), key=lambda x: x[1], reverse=True)
        msg = 'Number of missing values in each column:\n'
        for column, n in sort_by_nan_count:
            msg += f"  '{column}': {n} missing values\n"
        print(msg)
        return True

    def fill_missing_values(self):
        print(f'Fill in missing values by the mean of column.\n')
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns  # only numeric columns
        for column in numeric_columns:
            if self.df[column].isna().sum() > 0:
                self.df[column] = self.df[column].fillna(self.df[column].mean())


def is_binary(series: Iterable[Any]) -> bool:
    """
    A binary series should only contain 0 and 1. Anything else encountered will be considered as not binary.
    """
    for v in series:
        if pd.isna(v):
            continue  # ignore missing values
        if type(v) is str:
            return False
        elif type(v) is int or type(v) is float:
            if v != 0 and v != 1:
                return False
        else:
            return False
    return True


def summarize_numeric_outcome(df: pd.DataFrame, outcome: str):
    print(f'Outcome: "{outcome}"')  
    stats_str = df[outcome].describe().to_string()
    indented_stats = '\n'.join('- ' + line for line in stats_str.split('\n'))
    print(f'Summary statistics:\n{indented_stats}\n')


def summarize_binary_outcome(df: pd.DataFrame, outcome: str):
    print(f'Outcome: "{outcome}"')
    n_pos = df[outcome].value_counts()[1]
    n_neg = df[outcome].value_counts()[0]
    print(f'Positive(1): {n_pos} ({n_pos / len(df) * 100:.2f}%)')
    print(f'Negative(0): {n_neg} ({n_neg / len(df) * 100:.2f}%)\n')


def config_matplotlib_font_for_language(names: List[str]):
    matplotlib.rc('font', family='DejaVu Sans')  # default font for English
    for name in names:
        if contains_chinese(name):
            matplotlib.rc('font', family='Microsoft JhengHei')
            matplotlib.rc('axes', unicode_minus=False)  # show minus sign correctly when Chinese font is used
            break


def contains_chinese(s: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)
