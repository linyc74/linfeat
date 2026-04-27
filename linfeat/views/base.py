import pandas as pd
from typing import Any


def str_(value: Any) -> str:
    """
    Converts to str for GUI display
    """
    return '' if pd.isna(value) else str(value)
