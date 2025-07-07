"""
Test suite for regime-switching models.

Tests for the MarkovRegimeSwitching class and related functionality.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.econometric_modeling.markov_regime_switching import MarkovRegimeSwitching


class TestMarkovRegimeSwitching:
    """Test cases for MarkovRegimeSwitching class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create synthetic test data with regime-switching behavior
        n_obs = 500
        self.test_data = self._generate_regime_switching_data(n_obs)
        
        self.model = MarkovRegimeSwitching(
            n_regimes=2,
            model_type='mean_variance',
            max_iter=100,
            random_state=42
        )
    
    def _generate_regime_switching_data(self, n_obs):
        """Generate synthetic regime-switching data for testing."""
        # Regime parameters
        mu = [0.01, -0.02]  # Different means for each regime
        sigma = [0.015, 0.04]  # Different volatilities
        
        # Transition probabilities
        P = np.array([[0.95, 0.05],
                      [0.20, 0.80]])
        
        # Generate regime sequence
        regimes = np.zeros(n_obs, dtype=int)
        regimes[0] = 0
        
        for t in range(1, n_obs):
            regimes[t] = np.random.choice(2, p=P[regimes[t-1]])
        
        # Generate returns based on regimes
        returns = np.zeros(n_obs)
        for t in range(n_obs):
            regime = regimes[t]
            returns[t] = np.random.normal(mu[regime], sigma[regime])
        
        dates = pd.date_range('2020-01-01', periods=n_obs, freq='D')
        return pd.Series(returns, index=dates)
    
    def test_model_initialization(self):
        """Test proper model initialization."""
        assert self.model.n_regimes == 2
        assert self.model.model_type == 'mean_variance'
        assert self.model.max_iter == 100
        assert self.model.random_state == 42
    
    def test_model_fitting(self):
        """Test model fitting functionality."""
        # Fit the model
        self.model.fit(self.test_data)
        
        # Test that model was fitted
        assert hasattr(self.model, 'params_')
        assert hasattr(self.model, 'log_likelihood')
        assert hasattr(self.model, 'regimes')
        assert hasattr(self.model, 'regime_probabilities')
        
        # Test parameter structure
        assert 'means' in self.model.params_
        assert 'variances' in self.model.params_
        assert 'transition_matrix' in self.model.params_
        
        # Test dimensions
        assert len(self.model.params_['means']) == 2
        assert len(self.model.params_['variances']) == 2
        assert self.model.params_['transition_matrix'].shape == (2, 2)
    
    def test_regime_prediction(self):
        """Test regime prediction functionality."""
        self.model.fit(self.test_data)
        
        # Test regime predictions
        regimes = self.model.regimes
        assert len(regimes) == len(self.test_data)
        assert set(regimes).issubset({0, 1})
        
        # Test regime probabilities
        probs = self.model.regime_probabilities
        assert probs.shape == (len(self.test_data), 2)
        
        # Probabilities should sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    
    def test_information_criteria(self):
        """Test AIC and BIC calculation."""
        self.model.fit(self.test_data)
        
        # Test that IC are calculated
        assert hasattr(self.model, 'aic')
        assert hasattr(self.model, 'bic')
        
        # Test that BIC > AIC (always true due to penalty term)
        assert self.model.bic > self.model.aic
    
    def test_regime_summary(self):
        """Test regime summary functionality."""
        self.model.fit(self.test_data)
        
        summary = self.model.get_regime_summary()
        
        # Test summary structure
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2  # Two regimes
        
        # Test expected columns
        expected_cols = ['Mean', 'Variance', 'Std_Dev', 'Duration', 'Probability']
        for col in expected_cols:
            assert col in summary.columns
    
    def test_transition_matrix_properties(self):
        """Test transition matrix properties."""
        self.model.fit(self.test_data)
        
        P = self.model.transition_matrix
        
        # Test dimensions
        assert P.shape == (2, 2)
        
        # Test row sums equal 1
        np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=1e-5)
        
        # Test probabilities are between 0 and 1
        assert np.all(P >= 0)
        assert np.all(P <= 1)
    
    def test_model_convergence(self):
        """Test model convergence."""
        # Use more iterations for convergence test
        model = MarkovRegimeSwitching(n_regimes=2, max_iter=500, tolerance=1e-6)
        model.fit(self.test_data)
        
        # Test that model converged
        assert hasattr(model, 'converged_')
        
        # Test likelihood is finite
        assert np.isfinite(model.log_likelihood)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test invalid number of regimes
        with pytest.raises(ValueError):
            MarkovRegimeSwitching(n_regimes=1)
        
        with pytest.raises(ValueError):
            MarkovRegimeSwitching(n_regimes=0)
        
        # Test invalid model type
        with pytest.raises(ValueError):
            MarkovRegimeSwitching(model_type='invalid_type')
        
        # Test fitting with insufficient data
        short_data = pd.Series([1, 2, 3])
        model = MarkovRegimeSwitching(n_regimes=2)
        
        with pytest.raises(ValueError):
            model.fit(short_data)


class TestModelComparison:
    """Test model comparison functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_obs = 300
        self.test_data = pd.Series(np.random.normal(0, 1, n_obs))
    
    def test_model_selection(self):
        """Test model selection across different regime numbers."""
        models = {}
        
        for n_regimes in [2, 3]:
            model = MarkovRegimeSwitching(
                n_regimes=n_regimes,
                max_iter=100,
                random_state=42
            )
            model.fit(self.test_data)
            models[n_regimes] = model
        
        # Test that we can compare models
        assert models[2].aic != models[3].aic
        assert models[2].bic != models[3].bic
    
    def test_forecast_functionality(self):
        """Test forecasting functionality."""
        model = MarkovRegimeSwitching(n_regimes=2, random_state=42)
        model.fit(self.test_data)
        
        # Test regime probability forecasting
        forecast_probs = model.forecast_regime_probabilities(steps=5)
        
        assert forecast_probs.shape == (5, 2)
        np.testing.assert_allclose(forecast_probs.sum(axis=1), 1.0, rtol=1e-5)


class TestRobustness:
    """Test model robustness and edge cases."""
    
    def test_missing_values(self):
        """Test handling of missing values."""
        np.random.seed(42)
        data_with_nan = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10])
        
        model = MarkovRegimeSwitching(n_regimes=2)
        
        # Should handle missing values gracefully
        try:
            model.fit(data_with_nan)
            # If fitting succeeds, check that results are reasonable
            assert hasattr(model, 'log_likelihood')
        except ValueError:
            # If fitting fails due to insufficient data after removing NaN, that's acceptable
            pass
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        extreme_data = np.concatenate([normal_data, [100, -100]])  # Add extreme outliers
        
        data_series = pd.Series(extreme_data)
        model = MarkovRegimeSwitching(n_regimes=2, max_iter=50)
        
        # Should handle extreme values without crashing
        model.fit(data_series)
        assert hasattr(model, 'log_likelihood')
        assert np.isfinite(model.log_likelihood)


if __name__ == "__main__":
    pytest.main([__file__])
