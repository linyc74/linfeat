import os
import shutil
import pandas as pd
from typing import List
import statsmodels.api as sm
import statsmodels.formula.api as smf


class MultivariableRegression:

    df: pd.DataFrame
    variables: List[str]
    outcome: str
    outdir: str

    model: sm.OLS

    def main(self, df: pd.DataFrame, variables: List[str], outcome: str, outdir: str):
        self.df = df.copy()
        self.variables = variables
        self.outcome = outcome
        self.outdir = outdir

        os.makedirs(self.outdir, exist_ok=True)
        self.df = self.df[self.variables + [self.outcome]]

        self.fit_model()
        self.export_csv()
        self.export_txt()

    def fit_model(self):
        y = formula_format(self.outcome)
        x = ' + '.join([formula_format(c) for c in self.variables])
        formula = f'{y} ~ {x}'
        self.model = smf.ols(formula=formula, data=self.df).fit()

    def export_csv(self):
        ci = self.model.conf_int()
        ci.columns = ['95% CI Lower', '95% CI Upper']

        coef_df = pd.DataFrame({
            'Coefficient': self.model.params,
            'Std. Error': self.model.bse,
            't': self.model.tvalues,
            'P-value': self.model.pvalues,
            '95% CI Lower': ci['95% CI Lower'],
            '95% CI Upper': ci['95% CI Upper'],
        })

        # Rename index back to original variable names
        name_map = {}
        for var in self.variables:
            if not var.isidentifier():
                name_map[f'Q("{var}")'] = var

        new_index = []
        for idx in coef_df.index:
            if idx in name_map:
                new_index.append(name_map[idx])
            else:
                new_index.append(idx)
        coef_df.index = new_index
        coef_df.index.name = 'Variable'

        coef_df.to_csv(f'{self.outdir}/multivariable_regression.csv', encoding='utf-8-sig')

    def export_txt(self):
        lines = [
            f'OLS Regression Results',
            f'Outcome: {self.outcome}',
            f'',
            f'Model Fit Statistics',
            f'  N observations        : {int(self.model.nobs)}',
            f'  R-squared             : {self.model.rsquared:.4f}',
            f'  Adj. R-squared        : {self.model.rsquared_adj:.4f}',
            f'  F-statistic           : {self.model.fvalue:.4f}',
            f'  Prob (F-statistic)    : {self.model.f_pvalue:.4e}',
            f'  AIC                   : {self.model.aic:.4f}',
            f'  BIC                   : {self.model.bic:.4f}',
            f'  Log-Likelihood        : {self.model.llf:.4f}',
            f'  Df Model              : {int(self.model.df_model)}',
            f'  Df Residuals          : {int(self.model.df_resid)}',
            f'',
            f'Full Summary',
            f'',
            str(self.model.summary()),
        ]
        with open(f'{self.outdir}/multivariable_regression.txt', 'w') as f:
            f.write('\n'.join(lines))


def formula_format(s: str) -> str:
    """
    Return a patsy-safe (https://patsy.readthedocs.io) term.
    Bare name only if it is a valid Python identifier.
    Otherwise, wrap the name in quotes with Q(), e.g. Q("Blautia 菌屬").
    """
    if s.isidentifier():
        return s
    return f'Q("{s}")'
