import numpy as np
import pandas as pd
from test.setup import TestCase
from linfeat.model import cast_to_appropriate_type


class TestFunctions(TestCase):
    
    def test_cast_to_appropriate_type(self):
        actual = cast_to_appropriate_type('123')
        self.assertEqual(actual, 123)

        actual = cast_to_appropriate_type('123.456')
        self.assertEqual(actual, 123.456)
        
        actual = cast_to_appropriate_type(123)
        self.assertEqual(actual, 123)
        
        actual = cast_to_appropriate_type(123.456)
        self.assertEqual(actual, 123.456)
        
        actual = cast_to_appropriate_type(True)
        self.assertEqual(actual, True)
        
        actual = cast_to_appropriate_type(False)
        self.assertEqual(actual, False)
        
        actual = cast_to_appropriate_type(None)
        self.assertEqual(actual, None)
        
        actual = cast_to_appropriate_type(np.nan)
        self.assertTrue(pd.isna(actual))
        
        actual = cast_to_appropriate_type('')
        self.assertTrue(pd.isna(actual))

        actual = cast_to_appropriate_type('nan')
        self.assertTrue(pd.isna(actual))
        
        actual = cast_to_appropriate_type('abc')
        self.assertEqual(actual, 'abc')
        
        actual = cast_to_appropriate_type('True')  # do not convert to boolean
        self.assertEqual(actual, 'True')
        
        actual = cast_to_appropriate_type('False')  # do not convert to boolean
        self.assertEqual(actual, 'False')
