import pandas as pd
from test.setup import TestCase
from linfeat.multivariable import MultivariableRegression


class TestMultivariableRegression(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)

    # def tearDown(self):
    #     self.tear_down()

    def test_ols_regression(self):
        MultivariableRegression().main(
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
        )
