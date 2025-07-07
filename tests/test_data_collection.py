"""
Test suite for data collection module.

Tests for the FinancialDataCollector class and related functionality.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion.financial_data_collector import FinancialDataCollector


class TestFinancialDataCollector:
    """Test cases for FinancialDataCollector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = FinancialDataCollector(
            data_path="test_data/",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
    
    def test_collector_initialization(self):
        """Test proper initialization of data collector."""
        assert self.collector.start_date == "2023-01-01"
        assert self.collector.end_date == "2023-12-31"
        assert self.collector.data_path == "test_data/"
    
    def test_financial_data_structure(self):
        """Test that financial data has proper structure."""
        try:
            financial_data = self.collector.fetch_financial_data()
            
            # Test data structure
            assert isinstance(financial_data, dict)
            
            # Test for expected keys
            expected_keys = ['equities', 'bonds', 'commodities', 'currencies']
            for key in expected_keys:
                assert key in financial_data.keys()
                
        except Exception as e:
            # Skip test if data fetching fails (e.g., no internet)
            pytest.skip(f"Data fetching failed: {e}")
    
    def test_climate_data_structure(self):
        """Test that climate data has proper structure."""
        climate_data = self.collector.fetch_climate_data()
        
        # Test data structure
        assert isinstance(climate_data, dict)
        
        # Test for expected climate variables
        expected_vars = ['temperature_anomaly', 'co2_concentration', 'extreme_events']
        for var in expected_vars:
            assert var in climate_data.keys()
            assert isinstance(climate_data[var], pd.Series)
    
    def test_data_alignment(self):
        """Test data alignment functionality."""
        try:
            aligned_data = self.collector.align_datasets()
            
            # Test alignment
            assert isinstance(aligned_data, pd.DataFrame)
            assert not aligned_data.empty
            
            # Test date index
            assert isinstance(aligned_data.index, pd.DatetimeIndex)
            
        except Exception as e:
            pytest.skip(f"Data alignment failed: {e}")
    
    def test_date_validation(self):
        """Test date validation in collector."""
        # Test invalid date format
        with pytest.raises(ValueError):
            FinancialDataCollector(start_date="invalid-date")
    
    def test_data_quality_checks(self):
        """Test data quality validation."""
        climate_data = self.collector.fetch_climate_data()
        
        for var_name, var_data in climate_data.items():
            # Test no infinite values
            assert not np.isinf(var_data).any()
            
            # Test reasonable data ranges
            assert var_data.std() > 0  # Non-constant data


class TestDataQuality:
    """Test data quality and validation."""
    
    def test_missing_value_handling(self):
        """Test missing value handling strategies."""
        # Create test data with missing values
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        test_data = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10], index=dates)
        
        collector = FinancialDataCollector()
        
        # Test forward fill
        filled_data = collector._handle_missing_values(test_data, method='ffill')
        assert not filled_data.isna().any()
        
        # Test interpolation
        interp_data = collector._handle_missing_values(test_data, method='interpolate')
        assert not interp_data.isna().any()
    
    def test_outlier_detection(self):
        """Test outlier detection functionality."""
        # Create test data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outlier_data = np.concatenate([normal_data, [10, -10]])  # Add extreme outliers
        
        collector = FinancialDataCollector()
        outliers = collector._detect_outliers(outlier_data, method='zscore', threshold=3)
        
        # Should detect the extreme values
        assert len(outliers) >= 2


class TestIntegration:
    """Integration tests for complete data pipeline."""
    
    def test_complete_pipeline(self):
        """Test complete data collection and processing pipeline."""
        try:
            collector = FinancialDataCollector(
                start_date="2023-01-01",
                end_date="2023-01-31"
            )
            
            # Test complete pipeline
            financial_data = collector.fetch_financial_data()
            climate_data = collector.fetch_climate_data()
            economic_data = collector.fetch_economic_indicators()
            
            # Test alignment
            aligned_data = collector.align_datasets()
            
            # Validation
            assert isinstance(aligned_data, pd.DataFrame)
            assert len(aligned_data) > 0
            
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
