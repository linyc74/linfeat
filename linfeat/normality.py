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
    shapiro_p_threshold: float
    kolmogorov_p_threshold: float
    skewness_threshold: float
    excess_kurtosis_threshold: float
    outdir: str

    stats_data: List[Dict[str, float]]
    stats_df: pd.DataFrame

    def main(
            self,
            df: pd.DataFrame,
            variables: List[str],
            shapiro_p_threshold: float,
            kolmogorov_p_threshold: float,
            skewness_threshold: float,
            excess_kurtosis_threshold: float,
            outdir: str) -> pd.DataFrame:

        self.df = df
        self.variables = variables
        self.shapiro_p_threshold = shapiro_p_threshold
        self.kolmogorov_p_threshold = kolmogorov_p_threshold
        self.skewness_threshold = skewness_threshold
        self.excess_kurtosis_threshold = excess_kurtosis_threshold
        self.outdir = outdir

        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(f'{self.outdir}/QQ plot', exist_ok=True)

        self.stats_data = []
        for variable in self.variables:
            if type_of(df[variable]) != CONTINUOUS:
                print(f'Warning: Variable "{variable}" is not continuous. Skip normality test.')
                continue

            _, shapiro_p = shapiro(df[variable])
            _, kolmogorov_p = kstest(df[variable], 'norm')
            skewness = skew(df[variable], bias=False, nan_policy='omit')
            excess_kurtosis = kurtosis(df[variable], fisher=True, bias=False, nan_policy='omit')
            self.stats_data.append({
                'Variable': variable,
                'Shapiro-Wilk p-value': shapiro_p,
                'Kolmogorov-Smirnov p-value': kolmogorov_p,
                'Skewness': skewness,
                'Excess Kurtosis': excess_kurtosis,
            })
            self.qq_plot(variable)

        df = pd.DataFrame(self.stats_data)
        a = df['Shapiro-Wilk p-value'] > self.shapiro_p_threshold
        b = df['Kolmogorov-Smirnov p-value'] > self.kolmogorov_p_threshold
        c = df['Skewness'] <= self.skewness_threshold
        d = df['Excess Kurtosis'] <= self.excess_kurtosis_threshold
        df['Pass Normality Test'] = a & b & c & d
        self.stats_df = df

        self.stats_df.to_csv(f'{self.outdir}/normality.csv', encoding='utf-8-sig', index=False)
        self.write_summary()

        return self.stats_df

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
    
    def write_summary(self):
        passed = self.stats_df[self.stats_df['Pass Normality Test']]['Variable'].tolist()
        passed = [p.replace('\n', ' ') for p in passed]
        failed = self.stats_df[~self.stats_df['Pass Normality Test']]['Variable'].tolist()
        failed = [p.replace('\n', ' ') for p in failed]
        with open(f'{self.outdir}/summary.txt', 'w', encoding='utf-8-sig') as fh:
            fh.write(f'Shapiro-Wilk p-value threshold: {self.shapiro_p_threshold}\n')
            fh.write(f'Kolmogorov-Smirnov p-value threshold: {self.kolmogorov_p_threshold}\n')
            fh.write(f'Skewness threshold: {self.skewness_threshold}\n')
            fh.write(f'Excess Kurtosis threshold: {self.excess_kurtosis_threshold}\n\n')
            p_str = '\n- '.join(passed)
            f_str = '\n- '.join(failed)
            fh.write(f'The following {len(passed)} variables passed normality test:\n- {p_str}\n\n')
            fh.write(f'The following {len(failed)} variables failed normality test:\n- {f_str}\n')


def replace_invalid_path_chars(s: str) -> str:
    return s.replace('\\', '').replace('/', '|').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace('\n', '_')
