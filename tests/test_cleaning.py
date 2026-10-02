"""Tests for WDI data cleaning behavior."""
import pandas as pd
from src.data.cleaning import clean_indicators


def test_clean_indicators_removes_duplicate_and_casts():
    df = pd.DataFrame([
        {"country_code":"GBR","country":"United Kingdom","indicator_code":"X",
         "indicator_name":"Example","year":"2020","value":"10.5"},
        {"country_code":"GBR","country":"United Kingdom","indicator_code":"X",
         "indicator_name":"Example","year":"2020","value":"10.5"},
    ])
    out = clean_indicators(df)
    assert len(out) == 1
    assert int(out.iloc[0]["year"]) == 2020
    assert float(out.iloc[0]["value"]) == 10.5
