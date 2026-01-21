import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from statsmodels.stats.multitest import multipletests


P_VALUE_CORRECTION = 'fdr_bh'  # Benjamini-Hochberg
FDR = 0.05
COLORMAP = 'bwr_r'
LINEWIDTHS = 0.5
LINECOLOR = 'white'
COLOR_BAR_LABEL = 'Pearson correlation coefficient'
FIGSIZE = (22 / 2.54, 18 / 2.54)  # width, height (cm / 2.54)
DPI = 600
FONT_PROPERTIES = {
    'size': 7,
    'family': 'Microsoft JhengHei',
}
AXES_PROPERTIES = {
    'unicode_minus': False,  # show minus sign correctly when Chinese font is used
}


class PearsonMatrix:

    df: pd.DataFrame
    corr_df: pd.DataFrame
    p_value_df: pd.DataFrame

    def main(self, df: pd.DataFrame):
        self.df = df.copy()

        self.compute_pearson_correlation()
        self.correct_p_values()
        self.reorder_samples_by_hierarchical_clustering()
        self.mask_by_p_value()
        self.corr_df.to_csv(f'pearson_correlation_matrix.csv', encoding='utf-8-sig')
        self.p_value_df.to_csv(f'pearson_p_value_adjusted_matrix.csv', encoding='utf-8-sig')
        self.plot_heatmap()

    def compute_pearson_correlation(self):
        columns = self.df.columns
        corr_df = pd.DataFrame(index=columns, columns=columns)
        p_value_df = pd.DataFrame(index=columns, columns=columns)
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:  # diagonal = 1
                    corr_df.loc[col1, col2] = 1.0
                    p_value_df.loc[col1, col2] = 0.0
                else:
                    correlation, p_value = pearsonr(self.df[col1], self.df[col2])
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
        matplotlib.rc('font', **FONT_PROPERTIES)
        matplotlib.rc('axes', **AXES_PROPERTIES)
        plt.figure(figsize=FIGSIZE, dpi=DPI)
        sns.heatmap(
            self.corr_df,
            cmap=COLORMAP,
            vmax=1,
            vmin=-1,
            xticklabels=True,  # include every x label
            yticklabels=True,  # include every y label
            annot=False,
            linewidths=LINEWIDTHS,
            linecolor=LINECOLOR,
            cbar_kws={
                'label': COLOR_BAR_LABEL,
                'shrink': 0.5,
                'aspect': 20  # make color bar thinner
            }
        )
        plt.tight_layout()
        plt.savefig(f'pearson_correlation_matrix.png', dpi=DPI)
        plt.close()


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
