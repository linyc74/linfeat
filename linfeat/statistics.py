import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from statsmodels.stats.multitest import multipletests
from typing import List, Tuple, Dict, Any, Optional, Union
from scipy.stats.contingency import odds_ratio as scipy_odds_ratio
from scipy.stats import fisher_exact, chi2_contingency, ttest_ind, mannwhitneyu, f_oneway, kruskal, pearsonr, spearmanr
from .basic import determine_variable_type as type_of
from .basic import BINARY, CONTINUOUS, CATEGORICAL, config_matplotlib_font_for_language


class UnivariableStatistics:

    df: pd.DataFrame
    variables: List[str]
    outcome: str
    outdir: str
    parametric_outcome: bool
    parametric_features: List[str]
    colors: List[Tuple[float, float, float, float]]

    stats_data: List[Dict[str, Any]]
    stats_df: pd.DataFrame

    def main(
            self,
            df: pd.DataFrame,
            variables: List[str],
            outcome: str,
            outdir: str,
            parametric_outcome: bool,
            parametric_features: List[str],
            colors: str | List[str|Tuple[float, float, float, float]]):

        self.df = df.copy()
        self.variables = variables
        self.outcome = outcome
        self.outdir = outdir
        self.parametric_outcome = parametric_outcome
        self.parametric_features = parametric_features
        self.colors = get_colors(colors=colors)
        
        os.makedirs(self.outdir, exist_ok=True)

        self.reorder_variables()

        self.stats_data = []

        if type_of(self.df[self.outcome]) == BINARY:
            self.binary_outcome()
        
        elif type_of(self.df[self.outcome]) == CATEGORICAL:
            if len(self.df[self.outcome].dropna().unique()) == 2:
                self.binary_outcome()
            else:  # ≥ 3 categories
                raise ValueError(f'Categorical outcome "{self.outcome}" with more than 2 categories is not supported yet for univariable statistics.')

        elif type_of(self.df[self.outcome]) == CONTINUOUS:
            self.continuous_outcome()

        self.stats_df = pd.DataFrame(self.stats_data)
        self.correct_p_values()
        self.stats_df.to_csv(f'{self.outdir}/univariable_statistics.csv', encoding='utf-8-sig', index=False)

    def reorder_variables(self):
        # process all binary variables first, then categorical variables, then continuous variables
        def to_key(variable: str) -> int:
            var_type = type_of(self.df[variable])
            return {
                BINARY: 0,
                CATEGORICAL: 1,
                CONTINUOUS: 2,
            }[var_type]
        self.variables = sorted(self.variables, key=to_key)

    def binary_outcome(self):
        for feature in self.variables:
            if type_of(self.df[feature]) == BINARY:
                self.fisher_exact_test(x=feature, y=self.outcome)
            
            elif type_of(self.df[feature]) == CATEGORICAL:
                if len(self.df[feature].dropna().unique()) == 2:
                    self.fisher_exact_test(x=feature, y=self.outcome)
                else:  # ≥ 3 categories
                    self.chi_square_test(x=feature, y=self.outcome)
            
            elif type_of(self.df[feature]) == CONTINUOUS:
                if feature in self.parametric_features:
                    self.t_test(x=self.outcome, y=feature)
                else:
                    self.mann_whitney_u_test(x=self.outcome, y=feature)

    def continuous_outcome(self):
        for feature in self.variables:
            if type_of(self.df[feature]) == BINARY:
                if self.parametric_outcome:
                    self.t_test(x=feature, y=self.outcome)
                else:
                    self.mann_whitney_u_test(x=feature, y=self.outcome)
            
            elif type_of(self.df[feature]) == CATEGORICAL:
                if len(self.df[feature].dropna().unique()) == 2:
                    if self.parametric_outcome:
                        self.t_test(x=feature, y=self.outcome)
                    else:
                        self.mann_whitney_u_test(x=feature, y=self.outcome)
                else:  # ≥ 3 categories
                    if self.parametric_outcome:
                        self.anova(x=feature, y=self.outcome)
                    else:
                        self.kruskal_wallis_test(x=feature, y=self.outcome)
            
            elif type_of(self.df[feature]) == CONTINUOUS:
                if feature in self.parametric_features and self.parametric_outcome:  # both x and y needs to be parametric
                    self.pearson_correlation_test(x=feature, y=self.outcome)
                else:
                    self.spearman_correlation_test(x=feature, y=self.outcome)

    def fisher_exact_test(self, x: str, y: str):
        df = self.df.copy()[[x, y]].dropna(how='any')

        contingency_df = create_contingency_table(df=df, x=x, y=y)

        assert contingency_df.shape == (2, 2), f'Contingency table must be 2x2, but got {contingency_df.shape}'

        odds_ratio, ci_low, ci_high = calculate_odds_ratio(contingency_df)
        pvalue = fisher_exact(contingency_df).pvalue

        self.stats_data.append({
            'Statistical test': 'Fisher\'s exact test',
            'x': x,
            'y': y,
            'p-value': pvalue,
            'Odds ratio (OR)': odds_ratio,
            'OR 95% CI low': ci_low,
            'OR 95% CI high': ci_high,
        })

        outdir = f'{self.outdir}/Fisher\'s exact test'
        png = f'{x.replace('/', '|')} vs. {y.replace('/', '|')}.png'
        StackedBarPlot().main(
            count_df=contingency_df,
            colors=self.colors,
            title=format_(pvalue),
            png=f'{outdir}/{png}'
        )

    def chi_square_test(self, x: str, y: str):
        df = self.df.copy()[[x, y]].dropna(how='any')

        contingency_df = create_contingency_table(df=df, x=x, y=y)

        chi2, pvalue, dof, expected = chi2_contingency(contingency_df)
        
        self.stats_data.append({
            'Statistical test': 'Chi-square test',
            'x': x,
            'y': y,
            'p-value': pvalue,
            'Degrees of freedom': dof,
        })

        outdir = f'{self.outdir}/Chi-square test'
        png = f'{x.replace('/', '|')} vs. {y.replace('/', '|')}.png'
        StackedBarPlot().main(
            count_df=contingency_df,
            colors=self.colors,
            title=format_(pvalue),
            png=f'{outdir}/{png}'
        )

    def t_test(self, x: str, y: str):
        df = self.df.copy()[[x, y]].dropna(how='any')

        assert len(df[x].unique()) == 2, f'"{x}" is not binary or two-category'

        df.sort_values(by=x, inplace=True, ascending=True)

        group_0 = df[x] == df[x].unique()[0]
        group_1 = df[x] == df[x].unique()[1]

        vector_0 = df.loc[group_0, y]
        vector_1 = df.loc[group_1, y]

        statistic, pvalue = ttest_ind(vector_0, vector_1)

        self.stats_data.append({
            'Statistical test': 'Student\'s t-test',
            'x': x,
            'y': y,
            'p-value': pvalue,
            f'Mean ({df[x].unique()[0]})': vector_0.mean(),
            f'Mean ({df[x].unique()[1]})': vector_1.mean(),
        })

        outdir = f'{self.outdir}/Student\'s t-test'
        png = f'{x.replace('/', '|')} vs. {y.replace('/', '|')}.png'
        Boxplot().main(
            data=df,
            x=x,
            y=y,
            colors=self.colors,
            title=format_(pvalue),
            png=f'{outdir}/{png}'
        )

    def mann_whitney_u_test(self, x: str, y: str):
        df = self.df.copy()[[x, y]].dropna(how='any')

        assert len(df[x].unique()) == 2, f'"{x}" is not binary or two-category'

        df.sort_values(by=x, inplace=True, ascending=True)

        group_0 = df[x] == df[x].unique()[0]
        group_1 = df[x] == df[x].unique()[1]

        vector_0 = df.loc[group_0, y]
        vector_1 = df.loc[group_1, y]

        statistic, pvalue = mannwhitneyu(vector_0, vector_1)

        self.stats_data.append({
            'Statistical test': 'Mann-Whitney U test',
            'x': x,
            'y': y,
            'p-value': pvalue,
            f'Mean ({df[x].unique()[0]})': vector_0.mean(),
            f'Mean ({df[x].unique()[1]})': vector_1.mean(),
        })

        outdir = f'{self.outdir}/Mann-Whitney U test'
        png = f'{x.replace('/', '|')} vs. {y.replace('/', '|')}.png'
        Boxplot().main(
            data=df,
            x=x,
            y=y,
            colors=self.colors,
            title=format_(pvalue),
            png=f'{outdir}/{png}'
        )

    def anova(self, x: str, y: str):
        df = self.df.copy()[[x, y]].dropna(how='any')

        df.sort_values(by=x, inplace=True, ascending=True)

        group_names = self.df[x].unique()
        group_values = [self.df[y][self.df[x] == name] for name in group_names]
        group_means = [group.mean() for group in group_values]

        statistic, pvalue = f_oneway(*group_values)

        row = {
            'Statistical test': 'ANOVA',
            'x': x,
            'y': y,
            'p-value': pvalue,
        }
        for name, mean in zip(group_names, group_means):
            row[f'Mean ({name})'] = mean

        self.stats_data.append(row)
        
        outdir = f'{self.outdir}/ANOVA'
        png = f'{x.replace('/', '|')} vs. {y.replace('/', '|')}.png'
        Boxplot().main(
            data=self.df,
            x=x,
            y=y,
            colors=self.colors,
            title=format_(pvalue),
            png=f'{outdir}/{png}'
        )

    def kruskal_wallis_test(self, x: str, y: str):
        df = self.df.copy()[[x, y]].dropna(how='any')

        df.sort_values(by=x, inplace=True, ascending=True)

        group_names = self.df[x].unique()
        group_values = [self.df[y][self.df[x] == name] for name in group_names]
        group_means = [group.mean() for group in group_values]

        statistic, pvalue = kruskal(*group_values)

        row = {
            'Statistical test': 'Kruskal-Wallis Test',
            'x': x,
            'y': y,
            'p-value': pvalue,
        }
        for name, mean in zip(group_names, group_means):
            row[f'Mean ({name})'] = mean

        self.stats_data.append(row)

        outdir = f'{self.outdir}/Kruskal-Wallis Test'
        png = f'{x.replace('/', '|')} vs. {y.replace('/', '|')}.png'
        Boxplot().main(
            data=df,
            x=x,
            y=y,
            colors=self.colors,
            title=format_(pvalue),
            png=f'{outdir}/{png}'
        )

    def pearson_correlation_test(self, x: str, y: str):
        df = self.df.copy()[[x, y]].dropna(how='any')

        result = pearsonr(df[x], df[y])
        pvalue = result.pvalue

        self.stats_data.append({
            'Statistical test': 'Pearson Correlation Test',
            'x': x,
            'y': y,
            'p-value': pvalue,   
        })

        outdir = f'{self.outdir}/Pearson Correlation Test'
        png = f'{x.replace('/', '|')} vs. {y.replace('/', '|')}.png'
        ScatterPlot().main(
            data=df,
            x=x,
            y=y,
            colors=self.colors,
            title=format_(pvalue),
            png=f'{outdir}/{png}'
        )

    def spearman_correlation_test(self, x: str, y: str):
        df = self.df.copy()[[x, y]].dropna(how='any')

        result = spearmanr(df[x], df[y])
        pvalue = result.pvalue

        self.stats_data.append({
            'Statistical test': 'Spearman Correlation Test',
            'x': x,
            'y': y,
            'p-value': pvalue,
        })

        outdir = f'{self.outdir}/Spearman Correlation Test'
        png = f'{x.replace('/', '|')} vs. {y.replace('/', '|')}.png'
        ScatterPlot().main(
            data=df,
            x=x,
            y=y,
            colors=self.colors,
            title=format_(pvalue),
            png=f'{outdir}/{png}'
        )

    def correct_p_values(self):
        _, pvals_adjust, _, _ = multipletests(
            self.stats_df['p-value'].values,
            alpha=0.05,
            method='fdr_bh',  # Benjamini-Hochberg
            is_sorted=False,
            returnsorted=False)
        self.stats_df['p-adjust'] = pvals_adjust

        columns = self.stats_df.columns.tolist()
        p_pos = columns.index('p-value')
        columns.insert(p_pos + 1, 'p-adjust')  # 'p-value' and then 'p-adjust'
        columns = columns[:-1]  # remove the very last 'p-adjust' column
        self.stats_df = self.stats_df.reindex(columns=columns)


def create_contingency_table(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    df = df[[x, y]].copy()

    for var in [x, y]:
        assert df[var].notna().all(), f'"{var}" has missing values'

    for var in [x, y]:
        if type_of(df[var]) == BINARY:
            df[var] = df[var].astype(int).astype(str)  # 1.0 -> 1 -> '1'
        else:
            df[var] = df[var].astype(str)
        
    outdf = pd.DataFrame(
        columns=df[x].sort_values().unique(),
        index=df[y].sort_values().unique(),
        data=0,
        dtype=int,
    )
    outdf.columns.name = x
    outdf.index.name = y
    
    for _, row in df.iterrows():
        outdf.loc[row[y], row[x]] += 1
    
    return outdf


def calculate_odds_ratio(contingency_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
       x0 x1
    y0  a  b
    y1  c  d

    OR = (a/c) / (b/d) = (ad) / (bc)

    Returns (odds_ratio, ci_low, ci_high) at the 95% confidence level.
    Returns (nan, nan, nan) when the OR is undefined (zero cell in b or c).
    """
    df = contingency_df.copy()

    assert df.shape == (2, 2), f'Contingency table must be 2x2, but got {df.shape}'

    x0, x1 = df.columns.tolist()
    y0, y1 = df.index.tolist()

    a = df.loc[y0, x0]
    b = df.loc[y0, x1]
    c = df.loc[y1, x0]
    d = df.loc[y1, x1]

    if b == 0 or c == 0:
        return np.nan, np.nan, np.nan

    table = df.values.astype(int)
    result = scipy_odds_ratio(table, kind='sample')
    ci = result.confidence_interval(confidence_level=0.95)
    return result.statistic, ci.low, ci.high


def format_(pvalue: float) -> str:
    return f'$p < 0.001$' if pvalue < 0.001 else f'$p = {pvalue:.3f}$'


class StackedBarPlot:

    WIDTH = 5 / 2.54
    HEIGHT = 5 / 2.54
    FONT_SIZE = 6
    BAR_WIDTH = 0.6
    Y_LABEL = 'Count'

    df: pd.DataFrame
    colors: List[Tuple[float, float, float, float]]
    title: str
    png: str

    def main(
            self,
            count_df: pd.DataFrame,
            colors: List[Tuple[float, float, float, float]],
            title: str,
            png: str):

        self.df = count_df
        self.colors = colors
        self.title = title
        self.png = png

        self.init()
        self.plot()
        self.config()
        self.save()

    def init(self):
        matplotlib.rc('font', size=self.FONT_SIZE)
        config_matplotlib_font_for_language([self.df.columns.name, self.df.index.name])

        matplotlib.rc('axes', linewidth=0.5)

        outcome_name = self.df.index.name
        char_width = 0.1219  # cm at font size 7
        legend_width = len(outcome_name) * char_width / 2.54
        figsize = (self.WIDTH + legend_width, self.HEIGHT)

        plt.figure(figsize=figsize)

    def plot(self):
        bottom = np.zeros(shape=len(self.df.columns))
        for i in range(len(self.df.index)):
            plt.bar(
                x=self.df.columns,
                height=self.df.iloc[i, :],
                bottom=bottom,
                width=self.BAR_WIDTH,
                color=self.colors[i],
                edgecolor='black',
                linewidth=0.5
            )
            bottom += self.df.iloc[i, :]

    def config(self):
        plt.title(self.title, fontsize=self.FONT_SIZE)

        plt.xlabel(self.df.columns.name, fontsize=self.FONT_SIZE)
        plt.ylabel(self.Y_LABEL, fontsize=self.FONT_SIZE)

        plt.xlim(left=-1, right=len(self.df.columns))

        plt.gca().xaxis.set_tick_params(width=0.5)
        plt.gca().yaxis.set_tick_params(width=0.5)

        legend = plt.legend(self.df.index, title=self.df.index.name, bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.set_frame_on(False)

    def save(self):
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        os.makedirs(os.path.dirname(self.png), exist_ok=True)
        plt.savefig(self.png, dpi=600, bbox_inches='tight')
        plt.close()


class Boxplot:

    WIDTH_PADDING = 1.5 / 2.54
    WIDTH_PER_GROUP = 1.25 / 2.54
    HEIGHT = 5 / 2.54
    FONT_SIZE = 6
    BOX_WIDTH = 0.5

    data: pd.DataFrame
    x: str
    y: str
    colors: List[Tuple[float, float, float, float]]
    title: str
    png: str

    ax: matplotlib.axes.Axes

    def main(
            self,
            data: pd.DataFrame,
            x: str,
            y: str,
            colors: List[Tuple[float, float, float, float]],
            title: str,
            png: str):

        self.data = data
        self.x = x
        self.y = y
        self.colors = colors
        self.title = title
        self.png = png

        self.init()
        self.plot()
        self.config()
        self.save()

    def init(self):
        config_matplotlib_font_for_language([self.x, self.y])

        matplotlib.rc('font', size=self.FONT_SIZE)
        matplotlib.rc('axes', linewidth=0.5)

        groups = len(self.data[self.x].unique())
        figsize = (groups * self.WIDTH_PER_GROUP + self.WIDTH_PADDING, self.HEIGHT)

        plt.figure(figsize=figsize)

    def plot(self):
        self.ax = sns.boxplot(
            data=self.data,
            x=self.x,
            y=self.y,
            hue=self.x,
            palette=self.colors,
            width=self.BOX_WIDTH,
            linewidth=0.5,
            dodge=False,  # to align the boxes on the x axis
        )
        self.ax = sns.stripplot(
            data=self.data,
            x=self.x,
            y=self.y,
            hue=self.x,
            palette=self.colors,
            linewidth=0.25,
        )

    def config(self):
        self.ax.set_title(self.title, fontsize=self.FONT_SIZE)
        self.ax.set(xlabel=self.x, ylabel=self.y)
        plt.gca().xaxis.set_tick_params(width=0.5)
        plt.gca().yaxis.set_tick_params(width=0.5)
        plt.legend().remove()

    def save(self):
        plt.tight_layout()
        os.makedirs(os.path.dirname(self.png), exist_ok=True)
        plt.savefig(self.png, dpi=600)
        plt.close()


class ScatterPlot:

    WIDTH = 5 / 2.54
    HEIGHT = 5 / 2.54
    FONT_SIZE = 6
    POINT_SIZE = 18

    data: pd.DataFrame
    x: str
    y: str
    colors: List[Tuple[float, float, float, float]]
    title: str
    png: str

    ax: matplotlib.axes.Axes

    def main(
            self,
            data: pd.DataFrame,
            x: str,
            y: str,
            colors: List[Tuple[float, float, float, float]],
            title: str,
            png: str):

        self.data = data
        self.x = x
        self.y = y
        self.colors = colors
        self.title = title
        self.png = png

        self.init()
        self.plot()
        self.config()
        self.save()

    def init(self) -> None:
        config_matplotlib_font_for_language([self.x, self.y])

        matplotlib.rc('font', size=self.FONT_SIZE)
        matplotlib.rc('axes', linewidth=0.5)

        plt.figure(figsize=(self.WIDTH, self.HEIGHT))

    def plot(self) -> None:
        self.ax = sns.scatterplot(
            data=self.data,
            x=self.x,
            y=self.y,
            color=self.colors[0],
            edgecolor='black',
            linewidth=0.5,
            s=self.POINT_SIZE,
            alpha=0.85,
        )

    def config(self) -> None:
        self.ax.set_title(self.title, fontsize=self.FONT_SIZE)
        self.ax.set(xlabel=self.x, ylabel=self.y)
        plt.gca().xaxis.set_tick_params(width=0.5)
        plt.gca().yaxis.set_tick_params(width=0.5)

    def save(self) -> None:
        plt.tight_layout()
        os.makedirs(os.path.dirname(self.png), exist_ok=True)
        plt.savefig(self.png, dpi=600)
        plt.close()


def contains_chinese(s: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)


def get_colors(colors: str | List[str|Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:

    if isinstance(colors, str):  # is a colormap name
        cmap = plt.colormaps[colors]
        return [cmap(i) for i in range(len(colors))]

    elif isinstance(colors, list):  # is a list of color names, hex codes, or RGBA tuples
        return [to_rgba(c) for c in colors]
    
    else:
        raise ValueError(f'Invalid colors: "{colors}"')
