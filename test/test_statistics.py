import pandas as pd
from test.setup import TestCase
from linfeat.basic import Parameters
from linfeat.statistics import Statistics


class TestStatistics(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)
        
    def tearDown(self):
        self.tear_down()
    
    def test_main(self):
        parameters = Parameters()
        parameters.outdir = self.outdir
        Statistics().main(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            features=[
                'Binary Factor 1',
                '二元因子 2',
                'Binary Factor 3',
                'Parabacteroides_goldsteinii',
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Blautia 菌屬',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
            outcome='Obesity(1)/Normal(0)',
            parameters=parameters,
        )