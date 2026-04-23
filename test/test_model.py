import numpy as np
import pandas as pd
from test.setup import TestCase
from linfeat.model import cast_to_appropriate_type, cast_to_categorical


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
            print(f'value: {value}, expected: {expected}, actual: {actual}')
            print(f'type(value): {type(value)}, type(expected): {type(expected)}, type(actual): {type(actual)}')
            return
            if pd.isna(expected):
                self.assertTrue(pd.isna(actual))
            else:
                self.assertEqual(actual, expected)
                self.assertEqual(type(actual), type(expected))

    def test_cast_to_categorical(self):
        for value, expected in [
            (np.nan, np.nan),
            ('', np.nan),
            (None, np.nan),
            ('123', '123'),
            (123, '123'),
            (123.456, '123.456'),
            (True, 'True'),
            (False, 'False'),
            ('nan', 'nan'),
        ]:
            actual = cast_to_categorical(value)
            if pd.isna(expected):
                self.assertTrue(pd.isna(actual))
            else:
                self.assertEqual(actual, expected)
