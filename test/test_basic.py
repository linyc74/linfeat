import pandas as pd
from test.setup import TestCase
from linfeat.basic import PrepareData


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
