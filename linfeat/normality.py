import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from scipy.stats import shapiro, kstest, skew, kurtosis, probplot
from .basic import determine_variable_type as type_of
from .basic import config_matplotlib_font_for_language, CONTINUOUS


class Normality:

    df: pd.DataFrame
    variables: List[str]
    outdir: str

    stats_data: List[Dict[str, float]]

    def main(self, df: pd.DataFrame, variables: List[str], outdir: str):
        self.df = df
        self.variables = variables
        self.outdir = outdir

        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(f'{self.outdir}/QQ plot', exist_ok=True)

        self.stats_data = []
        for variable in self.variables:
            if type_of(self.df[variable]) != CONTINUOUS:
                print(f'Warning: Variable "{variable}" is not continuous. Skip normality test.')
                continue

            _, shapiro_p = shapiro(self.df[variable])
            _, ks_p = kstest(self.df[variable], 'norm')
            self.stats_data.append({
                'Variable': variable,
                'Kolmogorov-Smirnov p-value': ks_p,
                'Shapiro-Wilk p-value': shapiro_p,
                'Skewness': skew(self.df[variable]),
                'Kurtosis': kurtosis(self.df[variable]),
            })

            self.qq_plot(variable)

        pd.DataFrame(self.stats_data).to_csv(f'{self.outdir}/normality.csv', encoding='utf-8-sig', index=False)

    def qq_plot(self, variable: str):
        config_matplotlib_font_for_language([variable])
        matplotlib.rc('font', size=8)
        plt.figure(figsize=(8/2.54, 8/2.54), dpi=600)
        probplot(self.df[variable], dist='norm', plot=plt.gca())
        plt.title(variable)
        plt.xlabel('Theoretical quantiles')
        plt.ylabel('Sample quantiles')
        plt.tight_layout()
        plt.savefig(f'{self.outdir}/QQ plot/{replace_invalid_path_chars(variable)}.png', dpi=600)
        plt.close()


def replace_invalid_path_chars(s: str) -> str:
    return s.replace('\\', '').replace('/', '|').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace('\n', '_')
