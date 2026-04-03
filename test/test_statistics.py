import numpy as np
import pandas as pd
from test.setup import TestCase
from linfeat.statistics import UnivariableStatistics, create_contingency_table


class TestUnivariableStatistics(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)
        
    def tearDown(self):
        self.tear_down()
    
    def test_binary_outcome(self):
        UnivariableStatistics().main(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            variables=[
                'Blautia 菌屬',
                'Binary Factor 1',
                '二元因子 2',
                'Two Categories',
                'Five Categories',
                'Parabacteroides_goldsteinii',
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
            outcome='Obesity(1)/Normal(0)',
            outdir=self.outdir,
            parametric_features=[
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
        )

    def test_two_category_outcome(self):
        UnivariableStatistics().main(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            variables=[
                'Blautia 菌屬',
                'Binary Factor 1',
                '二元因子 2',
                'Obesity(1)/Normal(0)',
                'Five Categories',
                'Parabacteroides_goldsteinii',
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
            outcome='Two Categories',
            outdir=self.outdir,
            parametric_features=[
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
        )

    def test_more_than_two_category_outcome(self):
        with self.assertRaises(ValueError) as context:
            UnivariableStatistics().main(
                df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
                variables=['Blautia 菌屬'],
                outcome='Five Categories',
                outdir=self.outdir)
        self.assertEqual(str(context.exception), 'Categorical outcome "Five Categories" with more than 2 categories is not supported yet for univariable statistics.')

    def test_parametric_continuous_outcome(self):
        UnivariableStatistics().main(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            variables=[
                'Blautia 菌屬',
                'Binary Factor 1',
                '二元因子 2',
                'Obesity(1)/Normal(0)',
                'Two Categories',
                'Five Categories',
                'Parabacteroides_goldsteinii',
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
            ],
            outcome='Bacteroides',
            outdir=self.outdir,
            parametric_outcome=True,
            parametric_features=[
                'Faecalibaculum_rodentium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
            ],
        )

    def test_nonparametric_continuous_outcome(self):
        UnivariableStatistics().main(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            variables=[
                'Blautia 菌屬',
                'Binary Factor 1',
                '二元因子 2',
                'Obesity(1)/Normal(0)',
                'Two Categories',
                'Five Categories',
                'Parabacteroides_goldsteinii',
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
            ],
            outcome='Bacteroides',
            outdir=self.outdir,
            parametric_outcome=False,
            parametric_features=[
                'Faecalibaculum_rodentium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
            ],
        )


class TestCreateContingencyTable(TestCase):
    
    def test_no_na_allowed(self):
        df = pd.DataFrame(
            columns=['x', 'y'],
            data=[
                [0.0, np.nan]
            ]
        )
        with self.assertRaises(AssertionError):
            create_contingency_table(df=df, x='x', y='y')
    
    def test_2x2_binary(self):
        df = pd.DataFrame(
            columns=['x', 'y'],
            data=[
                [0.0, 0],
                [1.0, 0],
                [0.0, 1],
                [1.0, 1]
            ],
        )
        actual = create_contingency_table(df=df, x='x', y='y')
        expected = pd.DataFrame(
            columns=['0', '1'],
            index=['0', '1'],
            data=[
                [1, 1],
                [1, 1]
            ],
        )
        self.assertDataFrameEqual(actual, expected)
    
    def test_2x3_categorical_float(self):
        df = pd.DataFrame(
            columns=['x', 'y'],
            data=[
                [2.0, 0],
                [1.0, 0],
                [0.0, 1]
            ],
        )
        actual = create_contingency_table(df=df, x='x', y='y')
        expected = pd.DataFrame(
            columns=['0.0', '1.0', '2.0'],
            index=['0', '1'],
            data=[
                [0, 1, 1],
                [1, 0, 0]
            ],
        )
        self.assertDataFrameEqual(actual, expected)

    def test_2x2_categorical_str(self):
        df = pd.DataFrame(
            columns=['x', 'y'],
            data=[
                ['positive', 'positive'],
                ['negative', 'negative'],
            ],
        )
        actual = create_contingency_table(df=df, x='x', y='y')
        expected = pd.DataFrame(
            columns=['negative', 'positive'],
            index=['negative', 'positive'],
            data=[
                [1, 0],
                [0, 1]
            ],
        )
        self.assertDataFrameEqual(actual, expected)
