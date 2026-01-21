import os
import pandas as pd
from typing import List, Optional
from .matrix import CorrelationMatrix
from .linear import LinearL1FeatureSelection, LinearStepwiseFeatureSelection
from .logistic import LogisticL1FeatureSelection, LogisticStepwiseFeatureSelection
from .basic import Parameters, PrepareData, is_binary, summarize_numeric_outcome, summarize_binary_outcome


import builtins
from functools import partial
print = partial(builtins.print, flush=True)  # always flush the output


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

    for method in ['pearson', 'spearman']:
        CorrelationMatrix().main(
            df=df,
            parameters=parameters,
            method=method)

    core_features = [] if stepwise_core_features is None else stepwise_core_features

    if is_binary(df[outcome]):
        summarize_binary_outcome(
            df=df,
            outcome=outcome
        )
        LogisticL1FeatureSelection().main(
            df=df,
            features=features,
            outcome=outcome,
            parameters=parameters
        )
        LogisticStepwiseFeatureSelection().main(
            df=df,
            core_features=core_features,
            candidate_features=[c for c in df.columns if c not in [outcome] + core_features],
            outcome=outcome,
            parameters=parameters
        )
    else:
        summarize_numeric_outcome(
            df=df,
            outcome=outcome
        )
        LinearL1FeatureSelection().main(
            df=df,
            features=features,
            outcome=outcome,
            parameters=parameters
        )
        LinearStepwiseFeatureSelection().main(
            df=df,
            core_features=core_features,
            candidate_features=[c for c in df.columns if c not in [outcome] + core_features],
            outcome=outcome,
            parameters=parameters
        )


def linear_feature_selection(
        df: pd.DataFrame,
        features: List[str],
        outcome: str,
        stepwise_core_features: Optional[List[str]] = None,
        parameters: Optional[Parameters] = None):
    
    if parameters is None:
        parameters = Parameters()

    os.makedirs(parameters.outdir, exist_ok=True)

    summarize_numeric_outcome(df=df, outcome=outcome)

    df = PrepareData().main(df=df, features=features, outcome=outcome)
    
    LinearL1FeatureSelection().main(
        df=df,
        features=features,
        outcome=outcome,
        parameters=parameters
    )

    core_features=[] if stepwise_core_features is None else stepwise_core_features
    
    LinearStepwiseFeatureSelection().main(
        df=df,
        core_features=core_features,
        candidate_features=[c for c in df.columns if c not in [outcome] + core_features],
        outcome=outcome,
        parameters=parameters
    )


def logistic_feature_selection(
        df: pd.DataFrame,
        features: List[str],
        outcome: str,
        stepwise_core_features: Optional[List[str]] = None,
        parameters: Optional[Parameters] = None):

    if parameters is None:
        parameters = Parameters()

    os.makedirs(parameters.outdir, exist_ok=True)

    summarize_binary_outcome(df=df, outcome=outcome)

    df = PrepareData().main(df=df, features=features, outcome=outcome)
    
    LogisticL1FeatureSelection().main(
        df=df,
        features=features,
        outcome=outcome,
        parameters=parameters
    )

    core_features = [] if stepwise_core_features is None else stepwise_core_features

    LogisticStepwiseFeatureSelection().main(
        df=df,
        core_features=core_features,
        candidate_features=[c for c in df.columns if c not in [outcome] + core_features],
        outcome=outcome,
        parameters=parameters
    )

