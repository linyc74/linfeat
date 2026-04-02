import os
import pandas as pd
from typing import List, Optional
from .statistics import Statistics
from .matrix import CorrelationMatrix
from .linear import LinearL1FeatureSelection, LinearStepwiseFeatureSelection
from .logistic import LogisticL1FeatureSelection, LogisticStepwiseFeatureSelection
from .basic import Parameters, PrepareData, summarize_numeric_outcome, summarize_binary_outcome, determine_variable_type, BINARY, CONTINUOUS, CATEGORICAL


import builtins
from functools import partial
print = partial(builtins.print, flush=True)  # always flush the output


def univariable_statistics(df: pd.DataFrame, features: List[str], outcome: str, outdir: str):

    parameters = Parameters()
    parameters.outdir = outdir

    if determine_variable_type(df[outcome]) == BINARY:
        Statistics().main(
            df=df,
            features=features,
            outcome=outcome,
            parameters=parameters)


def linfeat(
        df: pd.DataFrame,
        features: List[str],
        outcome: str,
        stepwise_core_features: Optional[List[str]] = None,
        parameters: Optional[Parameters] = None):
    
    if parameters is None:
        parameters = Parameters()

    os.makedirs(parameters.outdir, exist_ok=True)

    df = PrepareData().main(df=df, features=features, outcome=outcome)
    df.to_csv(f'{parameters.outdir}/missing_values_filled_data.csv', encoding='utf-8-sig', index=False)

    for method in ['pearson', 'spearman']:
        CorrelationMatrix().main(df=df, parameters=parameters, method=method)

    core_features = [] if stepwise_core_features is None else stepwise_core_features

    if determine_variable_type(df[outcome]) == BINARY:
        summarize_binary_outcome(df=df, outcome=outcome)

        LogisticL1FeatureSelection().main(
            df=df,
            features=features,
            outcome=outcome,
            parameters=parameters)

        LogisticStepwiseFeatureSelection().main(
            df=df,
            core_features=core_features,
            candidate_features=[c for c in df.columns if c not in [outcome] + core_features],
            outcome=outcome,
            parameters=parameters)

    elif determine_variable_type(df[outcome]) == CONTINUOUS:
        summarize_numeric_outcome(df=df, outcome=outcome)

        LinearL1FeatureSelection().main(
            df=df,
            features=features,
            outcome=outcome,
            parameters=parameters)

        LinearStepwiseFeatureSelection().main(
            df=df,
            core_features=core_features,
            candidate_features=[c for c in df.columns if c not in [outcome] + core_features],
            outcome=outcome,
            parameters=parameters)

    elif determine_variable_type(df[outcome]) == CATEGORICAL:
        raise ValueError(f'Categorical outcome "{outcome}" is not supported yet for linear feature selection.')
