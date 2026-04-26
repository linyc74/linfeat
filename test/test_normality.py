import numpy as np
import pandas as pd
from test.setup import TestCase
from linfeat.normality import Normality


class TestNormality(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)
        
    def tearDown(self):
        self.tear_down()
    
    def test_main(self):
        df = pd.read_csv(f'{self.indir}/data.csv', index_col=0)
        Normality().main(
            df=df,
            variables=df.columns.tolist(),
            outdir=self.outdir,
        )
