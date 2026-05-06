import numpy as np
import pandas as pd
from test.setup import TestCase
from linfeat.model import cast_to_appropriate_type


class TestFunctions(TestCase):
    
    def test_cast_to_appropriate_type(self):
        for value, expected in [
            ('123', 123),
            ('123.456', 123.456),
            (123, 123),
            (123.456, 123.456),
            (123.0, 123),
            (True, True),
            (False, False),
            (None, np.nan),
            (np.nan, np.nan),
            ('', np.nan),
            ('nan', 'nan'),
            ('abc', 'abc'),
            ('True', 'True'),
            ('False', 'False'),
        ]:
            actual = cast_to_appropriate_type(value)
            if pd.isna(expected):
                self.assertTrue(pd.isna(actual))
            else:
                self.assertEqual(actual, expected)
                self.assertEqual(type(actual), type(expected))
