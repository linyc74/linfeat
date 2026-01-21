import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from statsmodels.stats.multitest import multipletests
from .basic import Parameters


P_VALUE_CORRECTION = 'fdr_bh'  # Benjamini-Hochberg
FDR = 0.05


class CorrelationMatrix:

    df: pd.DataFrame
    parameters: Parameters
    method: str

    corr_df: pd.DataFrame
    p_value_df: pd.DataFrame

    def main(self, df: pd.DataFrame, parameters: Parameters, method: str = 'pearson'):
        self.df = df.copy()
        self.parameters = parameters
        self.method = method

        self.compute_correlation()
        self.correct_p_values()
        self.reorder_samples_by_hierarchical_clustering()
        self.mask_by_p_value()
        self.corr_df.to_csv(f'{self.parameters.outdir}/{self.method}_correlation_matrix.csv', encoding='utf-8-sig')
        self.p_value_df.to_csv(f'{self.parameters.outdir}/{self.method}_p_value_adjusted_matrix.csv', encoding='utf-8-sig')
        self.plot_heatmap()

    def compute_correlation(self):

        if self.method == 'pearson':
            func = pearsonr
        elif self.method == 'spearman':
            func = spearmanr
        else:
            raise ValueError(f'Invalid correlation method: {self.method}')

        columns = self.df.columns
        corr_df = pd.DataFrame(index=columns, columns=columns)
        p_value_df = pd.DataFrame(index=columns, columns=columns)
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:  # diagonal = 1
                    corr_df.loc[col1, col2] = 1.0
                    p_value_df.loc[col1, col2] = 0.0
                else:
                    correlation, p_value = func(self.df[col1], self.df[col2])
                    corr_df.loc[col1, col2] = correlation
                    p_value_df.loc[col1, col2] = p_value

        # casting to float prevents very weird memory segmentation fault in matplotlib and seaborn
        # which could be caused by inconsistent float types in the correlation matrix
        self.corr_df = corr_df.astype(np.float64)
        self.p_value_df = p_value_df.astype(np.float64)

    def correct_p_values(self):
        n_features = len(self.df.columns)

        pvals = self.p_value_df.values.reshape(n_features ** 2)  # 2D -> 1D

        rejected, pvals_corrected, _, _ = multipletests(
            pvals,
            alpha=FDR,
            method=P_VALUE_CORRECTION,
            is_sorted=False,
            returnsorted=False)

        p_value_mat = pvals_corrected.reshape((n_features, n_features))  # 1D -> 2D

        self.p_value_df = pd.DataFrame(
            data=p_value_mat,
            index=self.df.columns,
            columns=self.df.columns)

    def reorder_samples_by_hierarchical_clustering(self):
        self.corr_df, self.p_value_df = ReorderSamplesByHierarchicalClustering().main(self.corr_df, self.p_value_df)

    def mask_by_p_value(self):
        self.corr_df = self.corr_df.mask(self.p_value_df > FDR).fillna(0)

    def plot_heatmap(self):
        matplotlib.rc('font', size=7)
        for name in self.df.columns:
            if contains_chinese(name):
                matplotlib.rc('font', family='Microsoft JhengHei')
                matplotlib.rc('axes', unicode_minus=False)  # show minus sign correctly when Chinese font is used
                break
        
        for with_color_bar in [True, False]:
            figsize = self.__get_figsize(with_color_bar=with_color_bar)
            plt.figure(figsize=figsize, dpi=600)
            sns.heatmap(
                self.corr_df,
                cmap='bwr_r',
                vmax=1,
                vmin=-1,
                xticklabels=True,  # include every x label
                yticklabels=True,  # include every y label
                annot=False,
                linewidths=0.5,
                linecolor='white',
                cbar=with_color_bar,
                cbar_kws={
                    'label': f'{self.method.capitalize()} correlation coefficient',
                    'shrink': 0.25,
                    'aspect': 15  # make color bar thinner
                }
            )
            plt.tight_layout()
            suffix = '_with_color_bar' if with_color_bar else ''
            plt.savefig(f'{self.parameters.outdir}/{self.method}_correlation_matrix{suffix}.png', dpi=600)
            plt.close()
    
    def __get_figsize(self, with_color_bar: bool) -> Tuple[float, float]:
        char_width = 0.1219  # cm at font size 7
        cell_size = 0.3341  # cm
        color_bar_width = 3.6 if with_color_bar else 0  # cm
        marginal_padding = 0.4  # cm
        
        n_features = len(self.df.columns)
        matrix_size = n_features * cell_size

        longest_feature_name = max(len(name) for name in self.df.columns)
        feature_name_padding = longest_feature_name * char_width
        
        width = matrix_size + feature_name_padding + marginal_padding + color_bar_width
        height = matrix_size + feature_name_padding + marginal_padding

        return width / 2.54, height / 2.54


def contains_chinese(s: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)


class ReorderSamplesByHierarchicalClustering:

    corr_df: pd.DataFrame
    p_value_df: pd.DataFrame

    def main(
            self,
            corr_df: pd.DataFrame,
            p_value_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        self.corr_df = corr_df
        self.p_value_df = p_value_df

        self.correct_correlation_matrix_values()
        self.reorder()

        return self.corr_df, self.p_value_df

    def correct_correlation_matrix_values(self):
        # needs to be diagonally symmetric
        # average to eliminate any floating point imprecision
        df = self.corr_df
        self.corr_df = (df + df.T) / 2

        # needs to be self identical, correlation must equal to 1
        # again to eliminate any floating point imprecision
        for i in range(self.corr_df.shape[0]):
            self.corr_df.iloc[i, i] = 1

    def reorder(self):  # this function is adapted from ChatGPT
        dist_mat = 1 - self.corr_df

        # convert the redundant n x n square matrix form into a condensed nC2 array
        # scipy's linkage function expects distances in this form.
        # since the distance matrix is symmetric, we can use squareform to convert it
        dist_array = squareform(dist_mat)

        # perform hierarchical clustering using Ward's method
        Z = linkage(dist_array, 'ward')

        # get the order of rows/columns as per the hierarchical clustering result
        leaf_order = leaves_list(Z)

        # reorder the distance matrix based on the clustering result
        self.corr_df = self.corr_df.iloc[leaf_order, leaf_order]
        self.p_value_df = self.p_value_df.iloc[leaf_order, leaf_order]
