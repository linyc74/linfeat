import pandas as pd
from test.setup import TestCase
from linfeat.basic import Parameters
from linfeat.workflows import linfeat


class TestWorkflows(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)
        
    def tearDown(self):
        self.tear_down()
    
    def test_linfeat_numeric_outcome(self):
        parameters = Parameters()
        parameters.outdir = self.outdir
        linfeat(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            features=[
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Blautia',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
            outcome='Parabacteroides_goldsteinii',
            parameters=parameters,
        )

    def test_linfeat_binary_outcome(self):
        parameters = Parameters()
        parameters.outdir = self.outdir
        linfeat(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            features=[
                'Parabacteroides_goldsteinii',
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Thomasclavelia',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Blautia',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
            outcome='Obesity(1)/Normal(0)',
            parameters=parameters,
        )
