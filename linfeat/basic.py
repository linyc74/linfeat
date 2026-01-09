import numpy as np
import pandas as pd
from typing import List


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
        has_missing_values = self.check_missing_values()
        if has_missing_values:
            self.fill_missing_values()

        return self.df

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
                self.df[column].fillna(self.df[column].mean(), inplace=True)
