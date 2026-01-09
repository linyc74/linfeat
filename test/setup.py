import os
import shutil
import unittest
import pandas as pd
from typing import Tuple
from os.path import relpath, dirname, join


def get_dirs(py_path: str) -> Tuple[str, str, str]:
    indir = relpath(path=py_path[:-3], start=os.getcwd())
    basedir = dirname(indir)
    workdir = join(basedir, 'workdir')
    outdir = join(basedir, 'outdir')
    return indir, workdir, outdir


class TestCase(unittest.TestCase):

    def set_up(self, py_path: str):
        self.indir, self.workdir, self.outdir = get_dirs(py_path=py_path)
        for d in [self.workdir, self.outdir]:
            os.makedirs(d, exist_ok=True)

    def tear_down(self):
        shutil.rmtree(self.workdir)
        shutil.rmtree(self.outdir)

    def assertDataFrameEqual(self, first: pd.DataFrame, second: pd.DataFrame):
        self.assertListEqual(list(first.columns), list(second.columns))
        self.assertListEqual(list(first.index), list(second.index))
        for c in first.columns:
            for i in first.index:
                a, b = first.loc[i, c], second.loc[i, c]
                if pd.isna(a) and pd.isna(b):
                    continue
                self.assertAlmostEqual(a, b)
