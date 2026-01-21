import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.stats import mannwhitneyu
from .basic import Parameters, is_binary


class Statistics:

    df: pd.DataFrame
    features: List[str]
    outcome: str
    parameters: Parameters

    def main(
            self,
            df: pd.DataFrame,
            features: List[str],
            outcome: str,
            parameters: Parameters):
        
        self.df = df
        self.features = features
        self.outcome = outcome
        self.parameters = parameters

        assert is_binary(self.df[self.outcome]), 'Outcome must be binary'
        
        for feature in self.features:
            if is_binary(self.df[feature]):
                ComputeStatisticsForBinaryFeature().main(
                    df=self.df,
                    feature=feature,
                    outcome=self.outcome,
                    parameters=self.parameters)
            else:
                ComputeStatisticsForNumericFeature().main(
                    df=self.df,
                    feature=feature,
                    outcome=self.outcome,
                    parameters=self.parameters)
    

class ComputeStatisticsForBinaryFeature:

    df: pd.DataFrame
    feature: str
    outcome: str
    parameters: Parameters

    def main(
            self,
            df: pd.DataFrame,
            feature: str,
            outcome: str,
            parameters: Parameters):
        
        self.df = df
        self.feature = feature
        self.outcome = outcome
        self.parameters = parameters

        self.df = self.df[self.df[self.feature].notna() & self.df[self.outcome].notna()]


class ComputeStatisticsForNumericFeature:
    
    BINARY_COLORS = [
        (0.8, 0.8, 0.8, 1.0),  # negative (0) light gray
        (0.4, 0.4, 0.4, 1.0),  # positive (1) dark gray
    ]
    
    df: pd.DataFrame
    feature: str
    outcome: str
    parameters: Parameters

    def main(
            self,
            df: pd.DataFrame,
            feature: str,
            outcome: str,
            parameters: Parameters):
        
        self.df = df
        self.feature = feature
        self.outcome = outcome
        self.parameters = parameters

        self.df = self.df[self.df[self.feature].notna() & self.df[self.outcome].notna()]
        self.df.sort_values(by=self.outcome, inplace=True)  # negative (0) first, positive (1) second

        negative = self.df[self.outcome] == 0
        positive = self.df[self.outcome] == 1

        statistic, pvalue = mannwhitneyu(
            x=self.df.loc[negative, self.feature],
            y=self.df.loc[positive, self.feature]
        )

        Boxplot().main(
            data=self.df,
            x=self.outcome,
            y=self.feature,
            colors=self.BINARY_COLORS,
            title=f'$p < 0.001$' if pvalue < 0.001 else f'$p = {pvalue:.3f}$',
            png=f'{self.parameters.outdir}/{pvalue:.4f}_{self.feature}.png'
        )


class Boxplot:

    WIDTH_PADDING = 1.5 / 2.54
    WIDTH_PER_GROUP = 1.25 / 2.54
    HEIGHT = 5 / 2.54
    DPI = 600
    FONT_SIZE = 6
    BOX_WIDTH = 0.5
    LINEWIDTH = 0.5
    BOX_lINEWIDTH = 0.5
    MARKER_LINEWIDTH = 0.25
    YLIM = None

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
        plt.rcParams['font.size'] = self.FONT_SIZE
        plt.rcParams['axes.linewidth'] = self.LINEWIDTH

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
            linewidth=self.BOX_lINEWIDTH,
            dodge=False,  # to align the boxes on the x axis
        )
        self.ax = sns.stripplot(
            data=self.data,
            x=self.x,
            y=self.y,
            hue=self.x,
            palette=self.colors,
            linewidth=self.MARKER_LINEWIDTH,
        )

    def config(self):
        self.ax.set_title(self.title)
        self.ax.set(xlabel=self.x, ylabel=self.y)
        plt.gca().xaxis.set_tick_params(width=self.LINEWIDTH)
        plt.gca().yaxis.set_tick_params(width=self.LINEWIDTH)
        plt.ylim(self.YLIM)
        plt.legend().remove()

    def save(self):
        plt.tight_layout()
        plt.savefig(self.png, dpi=self.DPI)
        plt.close()
