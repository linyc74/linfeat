import numpy as np
import pandas as pd
from test.setup import TestCase
from linfeat.basic import PrepareData, is_binary


class TestPrepareData(TestCase):

    def setUp(self):
        self.set_up(py_path=__file__)

    def tearDown(self):
        self.tear_down()
    
    def test_main(self):
        actual = PrepareData().main(
            df=pd.read_csv(f'{self.indir}/data.csv', index_col=0),
            features=['B', 'A'],
            outcome='Outcome',
        )
        expected = pd.read_csv(f'{self.indir}/result.csv', index_col=0)
        self.assertDataFrameEqual(actual, expected)


class TestFunctions(TestCase):

    def test_is_binary(self):
        self.assertTrue(is_binary([0, 1]))
        self.assertTrue(is_binary([0, 1, np.nan]))
        self.assertFalse(is_binary(['0', '1']))
        self.assertFalse(is_binary([0, 1, 2]))
