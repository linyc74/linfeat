import pandas as pd
from test.setup import TestCase
from linfeat.basic import Parameters
from linfeat.linear import LinearL1FeatureSelection, LinearStepwiseFeatureSelection


class TestLinearL1FeatureSelection(TestCase):
    
    def setUp(self):
        self.set_up(py_path=__file__)
        
    def tearDown(self):
        self.tear_down()
    
    def test_main(self):
        parameters = Parameters()
        parameters.outdir = self.outdir
        LinearL1FeatureSelection().main(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            features=[
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
            outcome='Parabacteroides_goldsteinii',
            parameters=parameters,
        )
    

class TestLinearStepwiseFeatureSelection(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)
        
    def tearDown(self):
        self.tear_down()
    
    def test_main(self):
        parameters = Parameters()
        parameters.outdir = self.outdir
        LinearStepwiseFeatureSelection().main(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            core_features=[],
            candidate_features=[
                'Muribaculum_intestinale',
                'Ligilactobacillus',
                'Lachnospiraceae',
                'Faecalibaculum_rodentium',
                'Blautia 菌屬',
                'Lachnoclostridium',
                'Erysipelotrichaceae',
                'Lactobacillus_johnsonii',
                'Bacteroides',
            ],
            outcome='Parabacteroides_goldsteinii',
            parameters=parameters,
        )
