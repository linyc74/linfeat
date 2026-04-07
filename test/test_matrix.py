import pandas as pd
from test.setup import TestCase
from linfeat.matrix import CorrelationMatrix


class TestCorrelationMatrix(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)
        
    def tearDown(self):
        self.tear_down()
    
    def test_pearson(self):
        df = pd.read_csv(f'{self.indir}/data_ch.csv', index_col=0)
        df['DM'] = 0  # should tolerate all-zero columns
        CorrelationMatrix().main(
            df=df,
            method='pearson',
            outdir=self.outdir,
        )

    def test_spearman(self):
        CorrelationMatrix().main(
            df=pd.read_csv(f'{self.indir}/data_en.csv', index_col=0),
            method='spearman',
            outdir=self.outdir,
        )
    
    def test_small_matrix(self):
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
            method='spearman',
            outdir=self.outdir,
        )
