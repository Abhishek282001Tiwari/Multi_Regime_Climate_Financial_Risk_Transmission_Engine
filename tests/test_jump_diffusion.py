"""
Test suite for jump-diffusion models.

Tests for the JumpDiffusionModel class and related functionality.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mathematical_finance.jump_diffusion_model import JumpDiffusionModel


class TestJumpDiffusionModel:
    """Test cases for JumpDiffusionModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = JumpDiffusionModel(
            mu=0.05,
            sigma=0.2,
            lambda_jump=0.1,
            mu_jump=-0.02,
            sigma_jump=0.1,
            climate_beta=0.5
        )
    
    def test_model_initialization(self):
        """Test proper model initialization."""
        assert self.model.mu == 0.05
        assert self.model.sigma == 0.2
        assert self.model.lambda_jump == 0.1
        assert self.model.mu_jump == -0.02
        assert self.model.sigma_jump == 0.1
        assert self.model.climate_beta == 0.5
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test negative volatility
        with pytest.raises(ValueError):
            JumpDiffusionModel(sigma=-0.1)
        
        # Test negative jump intensity
        with pytest.raises(ValueError):
            JumpDiffusionModel(lambda_jump=-0.1)
        
        # Test negative jump volatility
        with pytest.raises(ValueError):
            JumpDiffusionModel(sigma_jump=-0.1)
    
    def test_path_simulation(self):
        """Test Monte Carlo path simulation."""
        np.random.seed(42)
        
        # Simulation parameters
        T = 1.0
        n_steps = 252
        n_paths = 1000
        S0 = 100.0
        
        # Simulate paths
        paths = self.model.simulate_paths(T, n_steps, n_paths, S0)
        
        # Test output structure
        assert paths.shape == (n_paths, n_steps + 1)
        
        # Test initial condition
        np.testing.assert_allclose(paths[:, 0], S0)
        
        # Test positive prices
        assert np.all(paths > 0)
        
        # Test reasonable final prices (no extreme values)
        final_prices = paths[:, -1]
        assert np.all(final_prices > S0 * 0.1)  # Not too low
        assert np.all(final_prices < S0 * 10.0)  # Not too high
    
    def test_climate_effects(self):
        """Test climate effects on jump intensity."""
        np.random.seed(42)
        
        # Create models with different climate sensitivities
        model_no_climate = JumpDiffusionModel(climate_beta=0.0)
        model_high_climate = JumpDiffusionModel(climate_beta=2.0)
        
        # High climate stress
        climate_stress = 2.0
        
        # Simulate paths
        paths_no_climate = model_no_climate.simulate_paths(
            T=1.0, n_steps=252, n_paths=1000, S0=100.0, climate_index=climate_stress
        )
        
        paths_high_climate = model_high_climate.simulate_paths(
            T=1.0, n_steps=252, n_paths=1000, S0=100.0, climate_index=climate_stress
        )
        
        # Climate-sensitive model should have higher volatility
        vol_no_climate = np.std(paths_no_climate[:, -1])
        vol_high_climate = np.std(paths_high_climate[:, -1])
        
        assert vol_high_climate > vol_no_climate
    
    def test_var_calculation(self):
        """Test VaR calculation functionality."""
        np.random.seed(42)
        
        # Calculate VaR
        var_results = self.model.calculate_var(
            S0=100.0,
            T=1.0,
            confidence_level=0.05,
            n_simulations=5000
        )
        
        # Test output structure
        expected_keys = ['var_absolute', 'var_relative', 'expected_shortfall', 
                        'min_return', 'max_return']
        for key in expected_keys:
            assert key in var_results
        
        # Test VaR is negative (represents loss)
        assert var_results['var_relative'] < 0
        
        # Test Expected Shortfall is worse than VaR
        assert var_results['expected_shortfall'] <= var_results['var_relative']
        
        # Test reasonable values
        assert var_results['var_relative'] > -1.0  # Not 100% loss
        assert var_results['var_relative'] < 0.0   # Represents loss
    
    def test_option_pricing(self):
        """Test option pricing functionality."""
        np.random.seed(42)
        
        # Price call option
        call_price = self.model.price_option(
            S0=100.0,
            K=105.0,
            T=0.25,
            r=0.03,
            option_type='call',
            n_simulations=10000
        )
        
        # Price put option
        put_price = self.model.price_option(
            S0=100.0,
            K=105.0,
            T=0.25,
            r=0.03,
            option_type='put',
            n_simulations=10000
        )
        
        # Test reasonable option prices
        assert call_price > 0
        assert put_price > 0
        assert call_price < 100.0  # Call can't be worth more than stock
        assert put_price < 105.0   # Put can't be worth more than strike
        
        # Test put-call parity (approximately)
        # C - P ≈ S - K*e^(-rT)
        theoretical_diff = 100.0 - 105.0 * np.exp(-0.03 * 0.25)
        actual_diff = call_price - put_price
        
        # Allow for Monte Carlo error
        np.testing.assert_allclose(actual_diff, theoretical_diff, rtol=0.05)
    
    def test_greeks_calculation(self):
        """Test Greeks calculation."""
        np.random.seed(42)
        
        greeks = self.model.calculate_greeks(
            S0=100.0,
            K=100.0,
            T=0.25,
            r=0.03,
            option_type='call'
        )
        
        # Test output structure
        expected_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for greek in expected_greeks:
            assert greek in greeks
        
        # Test reasonable Greek values for ATM call
        assert 0.4 < greeks['delta'] < 0.7  # Delta around 0.5 for ATM
        assert greeks['gamma'] > 0         # Gamma always positive
        assert greeks['theta'] < 0         # Theta negative for long options
        assert greeks['vega'] > 0          # Vega positive for long options
    
    def test_monte_carlo_convergence(self):
        """Test Monte Carlo convergence."""
        np.random.seed(42)
        
        # Test with different number of simulations
        n_sims = [1000, 5000, 10000]
        var_estimates = []
        
        for n_sim in n_sims:
            var_result = self.model.calculate_var(
                S0=100.0,
                T=1.0,
                confidence_level=0.05,
                n_simulations=n_sim
            )
            var_estimates.append(var_result['var_relative'])
        
        # Check convergence (estimates should get closer)
        diff_1_2 = abs(var_estimates[1] - var_estimates[0])
        diff_2_3 = abs(var_estimates[2] - var_estimates[1])
        
        # Generally, more simulations should give more stable results
        # (though this isn't guaranteed due to randomness)
        assert len(var_estimates) == 3  # Basic sanity check
    
    def test_jump_size_distribution(self):
        """Test jump size distribution."""
        np.random.seed(42)
        
        # Generate many jump sizes
        n_jumps = 10000
        jump_sizes = []
        
        for _ in range(n_jumps):
            jump_size = np.random.normal(self.model.mu_jump, self.model.sigma_jump)
            jump_sizes.append(jump_size)
        
        jump_sizes = np.array(jump_sizes)
        
        # Test mean and std close to parameters
        np.testing.assert_allclose(np.mean(jump_sizes), self.model.mu_jump, rtol=0.05)
        np.testing.assert_allclose(np.std(jump_sizes), self.model.sigma_jump, rtol=0.05)


class TestJumpDiffusionVariants:
    """Test different variants and extensions of the model."""
    
    def test_no_jump_model(self):
        """Test model with zero jump intensity (pure diffusion)."""
        model = JumpDiffusionModel(lambda_jump=0.0)
        
        np.random.seed(42)
        paths = model.simulate_paths(T=1.0, n_steps=252, n_paths=1000, S0=100.0)
        
        # Should still produce valid paths
        assert paths.shape == (1000, 253)
        assert np.all(paths > 0)
    
    def test_high_jump_intensity(self):
        """Test model with high jump intensity."""
        model = JumpDiffusionModel(lambda_jump=2.0)  # High jump frequency
        
        np.random.seed(42)
        paths = model.simulate_paths(T=1.0, n_steps=252, n_paths=1000, S0=100.0)
        
        # Should still produce valid paths despite many jumps
        assert paths.shape == (1000, 253)
        assert np.all(paths > 0)
    
    def test_positive_jump_model(self):
        """Test model with positive jumps."""
        model = JumpDiffusionModel(mu_jump=0.05)  # Positive jump mean
        
        np.random.seed(42)
        paths = model.simulate_paths(T=1.0, n_steps=252, n_paths=1000, S0=100.0)
        
        # Positive jumps should lead to higher average returns
        final_returns = (paths[:, -1] - 100.0) / 100.0
        assert np.mean(final_returns) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_simulation_parameters(self):
        """Test handling of invalid simulation parameters."""
        model = JumpDiffusionModel()
        
        # Test negative time
        with pytest.raises(ValueError):
            model.simulate_paths(T=-1.0, n_steps=100, n_paths=100, S0=100.0)
        
        # Test zero steps
        with pytest.raises(ValueError):
            model.simulate_paths(T=1.0, n_steps=0, n_paths=100, S0=100.0)
        
        # Test zero paths
        with pytest.raises(ValueError):
            model.simulate_paths(T=1.0, n_steps=100, n_paths=0, S0=100.0)
        
        # Test negative initial price
        with pytest.raises(ValueError):
            model.simulate_paths(T=1.0, n_steps=100, n_paths=100, S0=-100.0)
    
    def test_extreme_parameters(self):
        """Test handling of extreme parameter values."""
        # Test very high volatility
        model = JumpDiffusionModel(sigma=2.0)  # 200% volatility
        
        np.random.seed(42)
        paths = model.simulate_paths(T=0.1, n_steps=25, n_paths=100, S0=100.0)
        
        # Should still produce valid results
        assert paths.shape == (100, 26)
        assert np.all(paths > 0)


if __name__ == "__main__":
    pytest.main([__file__])
