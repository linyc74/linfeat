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
            shapiro_p_threshold=0.05,
            kolmogorov_p_threshold=-1,
            skewness_threshold=1.0,
            excess_kurtosis_threshold=10.0,
            outdir=self.outdir,
        )
