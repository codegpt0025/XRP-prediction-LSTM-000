import pandas as pd
from src.data.unified_pipeline import DataIntelligence

def test_data_intelligence():
    """
    Tests the DataIntelligence component.
    """
    data_intelligence = DataIntelligence()

    # Test data collection
    data = data_intelligence.collect(asset='AAPL', lookback='1mo')

    # Check that the data is a pandas DataFrame
    assert isinstance(data, pd.DataFrame)

    # Check that the DataFrame is not empty
    assert not data.empty
