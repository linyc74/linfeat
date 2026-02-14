import pandas as pd
from test.setup import TestCase
from linfeat.basic import Parameters
from linfeat.matrix import CorrelationMatrix


class TestCorrelationMatrix(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)
        
    def tearDown(self):
        self.tear_down()
    
    def test_pearson(self):
        parameters = Parameters()
        parameters.outdir = self.outdir
        df = pd.read_csv(f'{self.indir}/data_ch.csv', index_col=0)
        df['DM'] = 0  # should tolerate all-zero columns
        CorrelationMatrix().main(
            df=df,
            parameters=parameters,
            method='pearson',
        )

    def test_spearman(self):
        parameters = Parameters()
        parameters.outdir = self.outdir
        CorrelationMatrix().main(
            df=pd.read_csv(f'{self.indir}/data_en.csv', index_col=0),
            parameters=parameters,
            method='spearman',
        )
    
    def test_small_matrix(self):
        parameters = Parameters()
        parameters.outdir = self.outdir
        CorrelationMatrix().main(
            df=pd.read_csv(f'{self.indir}/data_en.csv', index_col=0)[[
                'Pre-OP PA length',
                'Pre-OP PA width',
                'Pre-OP Fen',
                'Pre-OP Buccal fen',
                'Pre-OP Palatal or lingual fen',
                'Pre-OP Deh',
                'OP PA length',
                'OP PA width',
            ]],
            parameters=parameters,
            method='spearman',
        )
