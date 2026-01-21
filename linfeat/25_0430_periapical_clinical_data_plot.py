import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.axes
import matplotlib.pyplot as plt


class Main:

    CSV = '2025-04-30 case收集到100.csv'
    X_VARIABLE_TO_DATATYPE = {
        '年齡': float,
        'Systemic': str,
        'Free end': str,
        'Attrition': str,
        '對咬牙': str,
        'Restoration': str,
        '性別': str,
        '牙位': str,
        '醫師類別': str,
        '材料': str,
        'Pathology': str,
        '手術次數': int,
        '追蹤月份': float,

        # 術前評估
        '術前PA長': float,
        '術前PA寬': float,
        '術前囊袋深度': float,
        '術前Probing位置': float,
        '術前Fen': str,
        '術前Deh': str,
        '術前Thru': str,

        # 術中評估
        '術中Prep長': float,
        '術中Prep寬': float,
        '術中PA長': float,
        '術中PA寬': float,
        '術中GTR': str,
        '術中Fen': str,
        '術中Deh': str,
        '術中Thru': str,
    }
    Y_VARIABLE_TO_DATATYPE = {
        # 術後評估
        'MOLVEN': str,
    }
    OUTDIR = './plots'

    df: pd.DataFrame

    def main(self):
        self.df = pd.read_csv(self.CSV)
        self.cast_datatypes()
        os.makedirs(self.OUTDIR, exist_ok=True)

        for x, x_dtype in self.X_VARIABLE_TO_DATATYPE.items():
            for y, y_dtype in self.Y_VARIABLE_TO_DATATYPE.items():

                x_is_numeric = x_dtype in [float, int]
                y_is_numeric = y_dtype in [float, int]

                if x_is_numeric and y_is_numeric:
                    self.scatterplot(x=x, y=y)

                elif (not x_is_numeric) and (not y_is_numeric):  # both categorical
                    self.stacked_count_barplot(x=x, y=y)

                else:  # one of x and y is categorical, the other numeric
                    self.boxplot(x=x, y=y)

    def cast_datatypes(self):
        for var_to_dtype in [
            self.X_VARIABLE_TO_DATATYPE,
            self.Y_VARIABLE_TO_DATATYPE,
        ]:
            for var, dtype in var_to_dtype.items():
                self.df[var] = self.df[var].astype(dtype)

    def scatterplot(self, x: str, y: str):
        ScatterPlot().main(
            data=self.df,
            x=x,
            y=y,
            png=f'{self.OUTDIR}/{x} vs {y}.png'
        )

    def stacked_count_barplot(self, x: str, y: str):
        count_df = CreateCountDataFrame().main(
            df=self.df,
            x=x,
            y=y
        )
        StackedBarPlot().main(
            count_df=count_df,
            png=f'{self.OUTDIR}/{x} vs {y}.png'
        )

    def boxplot(self, x: str, y: str):
        data = self.df.sort_values(
            by=y,
            ascending=True
        )
        BoxPlot().main(
            data=data,
            x=x,
            y=y,
            png=f'{self.OUTDIR}/{x} vs {y}.png'
        )


class ScatterPlot:

    data: pd.DataFrame
    x: str
    y: str
    png: str

    def main(
            self,
            data: pd.DataFrame,
            x: str,
            y: str,
            png: str):

        self.data = data
        self.x = x
        self.y = y
        self.png = png

        raise NotImplementedError # to be implemented in the future


class CreateCountDataFrame:

    df: pd.DataFrame
    x: str
    y: str

    outdf: pd.DataFrame

    def main(
            self,
            df: pd.DataFrame,
            x: str,
            y: str) -> pd.DataFrame:

        self.df = df
        self.x = x
        self.y = y

        self.set_empty_outdf()

        for i, row in self.df.iterrows():
            which_y = row[self.y].strip()
            which_x = row[self.x].strip()
            self.outdf.loc[which_y, which_x] += 1

        return self.outdf

    def set_empty_outdf(self):
        columns = []
        for item in self.df[self.x]:
            if item.strip() not in columns:
                columns.append(item.strip())

        indexes = []
        for item in self.df[self.y]:
            if item.strip() not in indexes:
                indexes.append(item.strip())

        self.outdf = pd.DataFrame(
            columns=sorted(columns),
            index=sorted(indexes),
            data=0)

        self.outdf.columns.name = self.x
        self.outdf.index.name = self.y


class StackedBarPlot:

    FIGSIZE = (8 / 2.54, 6 / 2.54)
    DPI = 600
    FONT = 'Microsoft JhengHei'
    FONT_SIZE = 8
    LINEWIDTH = 0.5
    BAR_WIDTH = 0.75
    COLORS = [
        'lightblue',
        # 'deepskyblue',
        # 'dodgerblue',
        'royalblue',
        'mediumblue',
        # 'darkblue',
        'midnightblue',
        # 'navy',
    ]
    Y_LABEL = 'Count'

    df: pd.DataFrame
    png: str

    def main(
            self,
            count_df: pd.DataFrame,
            png: str):

        self.df = count_df
        self.png = png

        self.init()
        self.plot()
        self.config()
        self.save()

    def init(self):
        plt.rcParams['font.size'] = self.FONT_SIZE
        plt.rcParams['axes.linewidth'] = self.LINEWIDTH
        plt.rcParams['font.family'] = self.FONT
        plt.figure(figsize=self.FIGSIZE)

    def plot(self):
        bottom = np.zeros(shape=len(self.df.columns))
        for i in range(len(self.df.index)):
            plt.bar(
                x=self.df.columns,
                height=self.df.iloc[i, :],
                bottom=bottom,
                width=self.BAR_WIDTH,
                color=self.COLORS[i],
                edgecolor='black',
                linewidth=self.LINEWIDTH
            )
            bottom += self.df.iloc[i, :]

    def config(self):
        plt.xlabel(self.df.columns.name, fontsize=self.FONT_SIZE)
        plt.ylabel(self.Y_LABEL, fontsize=self.FONT_SIZE)

        plt.xlim(left=-1, right=len(self.df.columns))

        plt.xticks(rotation=90)
        plt.gca().xaxis.set_tick_params(width=self.LINEWIDTH)
        plt.gca().yaxis.set_tick_params(width=self.LINEWIDTH)

        legend = plt.legend(self.df.index, title=self.df.index.name, bbox_to_anchor=(1, 1))
        legend.set_frame_on(False)

    def save(self):
        plt.tight_layout()
        plt.savefig(self.png, dpi=self.DPI)
        plt.close()


class BoxPlot:

    FIGSIZE = (8 / 2.54, 6 / 2.54)
    DPI = 600
    FONT = 'Microsoft JhengHei'
    FONT_SIZE = 8
    BOX_WIDTH = 0.5
    LINEWIDTH = 0.5
    BOX_lINEWIDTH = 0.5
    MARKER_LINEWIDTH = 0.25
    COLORS = [  # or palette, e.g. 'Set1'
        'lightblue',
        # 'deepskyblue',
        # 'dodgerblue',
        'royalblue',
        'mediumblue',
        # 'darkblue',
        'midnightblue',
        # 'navy',
    ]

    data: pd.DataFrame
    x: str
    y: str
    png: str

    ax: matplotlib.axes.Axes

    def main(
            self,
            data: pd.DataFrame,
            x: str,
            y: str,
            png: str):

        self.data = data
        self.x = x
        self.y = y
        self.png = png

        self.init()
        self.plot()
        self.config()
        self.save()

    def init(self):
        plt.rcParams['font.size'] = self.FONT_SIZE
        plt.rcParams['axes.linewidth'] = self.LINEWIDTH
        plt.rcParams['font.family'] = self.FONT
        plt.figure(figsize=self.FIGSIZE)

    def plot(self):
        self.ax = sns.boxplot(
            data=self.data,
            x=self.x,
            y=self.y,
            hue=self.y,
            legend=False,
            palette=self.COLORS,
            width=self.BOX_WIDTH,
            linewidth=self.BOX_lINEWIDTH,
            dodge=False,  # to align the boxes on the x axis
        )
        self.ax = sns.stripplot(
            data=self.data,
            x=self.x,
            y=self.y,
            hue=self.y,
            legend=False,
            palette=self.COLORS,
            linewidth=self.MARKER_LINEWIDTH,
        )

    def config(self):
        plt.gca().xaxis.set_tick_params(width=self.LINEWIDTH)
        plt.gca().yaxis.set_tick_params(width=self.LINEWIDTH)

    def save(self):
        plt.tight_layout()
        plt.savefig(self.png, dpi=self.DPI)
        plt.close()


if __name__ == '__main__':
    Main().main()
