import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_all_configs  # noqa: E402
from src.data.preprocess import preprocess  # noqa: E402
from src.environment.workload import generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def cfgs():
    c = load_all_configs()
    c = copy.deepcopy(c)
    c["data"]["synthetic"]["n_services"] = 2
    c["data"]["synthetic"]["n_days"] = 10
    return c


@pytest.fixture(scope="session")
def raw_df(cfgs):
    return generate_dataset(cfgs["data"], cfgs["simulation"])


@pytest.fixture(scope="session")
def processed(cfgs, raw_df):
    df, meta = preprocess(raw_df, cfgs["data"])
    return df, meta
