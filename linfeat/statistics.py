import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Tuple, Dict, Any
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests
from .basic import Parameters, is_binary, config_matplotlib_font_for_language


BINARY_OUTCOME_COLORS = [
    (0.8, 0.8, 0.8, 1.0),  # negative (0) light gray
    (0.4, 0.4, 0.4, 1.0),  # positive (1) dark gray
]


class Statistics:

    df: pd.DataFrame
    features: List[str]
    outcome: str
    parameters: Parameters

    stats_data: List[Dict[str, Any]]
    stats_df: pd.DataFrame

    def main(
            self,
            df: pd.DataFrame,
            features: List[str],
            outcome: str,
            parameters: Parameters):
        
        self.df = df
        self.features = features
        self.outcome = outcome
        self.parameters = deepcopy(parameters)

        os.makedirs(self.parameters.outdir, exist_ok=True)

        assert is_binary(self.df[self.outcome]), 'Outcome must be binary'

        self.stats_data = []
        for feature in self.features:
            if is_binary(self.df[feature]):
                self.binary_feature(feature)
            else:
                self.numeric_feature(feature)

        self.stats_df = pd.DataFrame(self.stats_data)
        self.correct_p_values()
        self.stats_df.to_csv(f'{self.parameters.outdir}/statistics.csv', encoding='utf-8-sig', index=False)
        
    def binary_feature(self, feature: str):
        self.df = self.df[self.df[feature].notna() & self.df[self.outcome].notna()]

        contingency_df = create_contingency_table(df=self.df, x=feature, y=self.outcome)
        pvalue = fisher_exact(contingency_df).pvalue
        self.stats_data.append({
            'Feature': feature,
            'Feature type': 'Binary',
            'Statistic': 'Fisher\'s Exact Test',
            'P value': pvalue,
        })
        outdir = f'{self.parameters.outdir}/binary_features'
        os.makedirs(outdir, exist_ok=True)
        StackedBarPlot().main(
            count_df=contingency_df,
            title=f'$p < 0.001$' if pvalue < 0.001 else f'$p = {pvalue:.3f}$',
            png=f'{outdir}/{feature}.png'
        )
    
    def numeric_feature(self, feature: str):
        self.df = self.df[self.df[feature].notna() & self.df[self.outcome].notna()]

        self.df.sort_values(by=self.outcome, inplace=True, ascending=True)  # negative (0) first, positive (1) second

        negative = self.df[self.outcome] == 0
        positive = self.df[self.outcome] == 1

        statistic, pvalue = mannwhitneyu(
            x=self.df.loc[negative, feature],
            y=self.df.loc[positive, feature]
        )
        self.stats_data.append({
            'Feature': feature,
            'Feature type': 'Numeric',
            'Statistic': 'Mann-Whitney U test',
            'P value': pvalue,
        })

        outdir = f'{self.parameters.outdir}/numeric_features'
        os.makedirs(outdir, exist_ok=True)
        Boxplot().main(
            data=self.df,
            x=self.outcome,
            y=feature,
            colors=BINARY_OUTCOME_COLORS,
            title=f'$p < 0.001$' if pvalue < 0.001 else f'$p = {pvalue:.3f}$',
            png=f'{outdir}/{feature}.png'
        )

    def correct_p_values(self):
        _, pvals_adjusted, _, _ = multipletests(
            self.stats_df['P value'].values,
            alpha=0.05,
            method='fdr_bh',  # Benjamini-Hochberg
            is_sorted=False,
            returnsorted=False)
        self.stats_df['P adjusted'] = pvals_adjusted


def create_contingency_table(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    if df[x].dtype == float:
        df[x] = df[x].astype(int)
    if df[y].dtype == float:
        df[y] = df[y].astype(int)
        
    outdf = pd.DataFrame(
        columns=[0, 1],
        index=[0, 1],
        data=0,
        dtype=int,
    )
    outdf.columns.name = x
    outdf.index.name = y
    
    for i, row in df.iterrows():
        which_y = row[y]
        which_x = row[x]
        outdf.loc[which_y, which_x] += 1
    
    assert outdf.shape == (2, 2)  # contingency table must be 2x2
    
    return outdf


class StackedBarPlot:

    WIDTH = 5 / 2.54
    HEIGHT = 5 / 2.54
    FONT_SIZE = 6
    BAR_WIDTH = 0.6
    Y_LABEL = 'Count'

    df: pd.DataFrame
    title: str
    png: str

    def main(
            self,
            count_df: pd.DataFrame,
            title: str,
            png: str):

        self.df = count_df
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
                color=BINARY_OUTCOME_COLORS[i],
                edgecolor='black',
                linewidth=0.5
            )
            bottom += self.df.iloc[i, :]

    def config(self):
        plt.title(self.title, fontsize=self.FONT_SIZE)

        plt.xlabel(self.df.columns.name, fontsize=self.FONT_SIZE)
        plt.ylabel(self.Y_LABEL, fontsize=self.FONT_SIZE)

        plt.xlim(left=-1, right=len(self.df.columns))

        plt.xticks(rotation=90)
        plt.gca().xaxis.set_tick_params(width=0.5)
        plt.gca().yaxis.set_tick_params(width=0.5)

        legend = plt.legend(self.df.index, title=self.df.index.name, bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.set_frame_on(False)

    def save(self):
        plt.tight_layout(rect=[0, 0, 0.85, 1])
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
        plt.savefig(self.png, dpi=600)
        plt.close()


def contains_chinese(s: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)
