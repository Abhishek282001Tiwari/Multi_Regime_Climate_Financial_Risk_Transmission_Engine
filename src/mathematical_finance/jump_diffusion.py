"""
Jump-Diffusion Models with Climate Triggers
Implements jump-diffusion processes with climate-triggered jumps.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import logging

class JumpDiffusionModel:
    """Jump-diffusion model with climate-triggered jumps."""
    
    def __init__(self, mu: float = 0.05, sigma: float = 0.2, 
                 jump_intensity: float = 0.1, jump_mean: float = -0.05, 
                 jump_std: float = 0.1):
        """
        Initialize jump-diffusion model.
        
        Args:
            mu: Drift parameter
            sigma: Diffusion volatility
            jump_intensity: Jump intensity (lambda)
            jump_mean: Mean jump size
            jump_std: Standard deviation of jump sizes
        """
        self.mu = mu
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.logger = logging.getLogger(__name__)
        
    def simulate_path(self, T: float, steps: int, S0: float = 100.0,
                     climate_events: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate a jump-diffusion path.
        
        Args:
            T: Time horizon
            steps: Number of time steps
            S0: Initial value
            climate_events: Array of climate event indicators
            
        Returns:
            Simulated path
        """
        dt = T / steps
        t = np.linspace(0, T, steps + 1)
        
        # Initialize path
        S = np.zeros(steps + 1)
        S[0] = S0
        
        # Generate random numbers
        np.random.seed(42)  # For reproducibility
        dW = np.random.normal(0, np.sqrt(dt), steps)  # Brownian increments
        
        for i in range(steps):
            # Diffusion component
            diffusion = self.mu * dt + self.sigma * dW[i]
            
            # Jump component
            jump = 0.0
            
            # Climate-triggered jumps
            if climate_events is not None and i < len(climate_events):
                climate_intensity = climate_events[i] * self.jump_intensity * 2  # Amplify during climate events
            else:
                climate_intensity = self.jump_intensity
            
            # Generate Poisson jumps
            n_jumps = np.random.poisson(climate_intensity * dt)
            
            if n_jumps > 0:
                jump_sizes = np.random.normal(self.jump_mean, self.jump_std, n_jumps)
                jump = np.sum(jump_sizes)
            
            # Update path
            S[i + 1] = S[i] * np.exp(diffusion + jump)
        
        return S
    
    def simulate_multiple_paths(self, T: float, steps: int, n_paths: int, 
                               S0: float = 100.0, 
                               climate_events: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate multiple jump-diffusion paths.
        
        Args:
            T: Time horizon
            steps: Number of time steps
            n_paths: Number of paths to simulate
            S0: Initial value
            climate_events: Array of climate event indicators
            
        Returns:
            Array of simulated paths (n_paths x steps+1)
        """
        paths = np.zeros((n_paths, steps + 1))
        
        for i in range(n_paths):
            paths[i] = self.simulate_path(T, steps, S0, climate_events)
        
        return paths
    
    def calculate_option_price_mc(self, S0: float, K: float, T: float, r: float,
                                 option_type: str = 'call', n_simulations: int = 100000,
                                 climate_events: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate option price using Monte Carlo simulation.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            n_simulations: Number of simulations
            climate_events: Array of climate event indicators
            
        Returns:
            Dictionary with option price and statistics
        """
        steps = 252  # Daily steps for one year
        if T < 1:
            steps = int(252 * T)
        
        # Simulate paths
        paths = self.simulate_multiple_paths(T, steps, n_simulations, S0, climate_events)
        
        # Calculate payoffs
        final_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - final_prices, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        
        # Discount payoffs
        discounted_payoffs = payoffs * np.exp(-r * T)
        
        # Calculate statistics
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        return {
            'option_price': option_price,
            'std_error': std_error,
            'confidence_interval_95': [
                option_price - 1.96 * std_error,
                option_price + 1.96 * std_error
            ],
            'payoff_mean': np.mean(payoffs),
            'payoff_std': np.std(payoffs)
        }
    
    def calculate_greeks(self, S0: float, K: float, T: float, r: float,
                        option_type: str = 'call', n_simulations: int = 100000,
                        climate_events: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: Option type
            n_simulations: Number of simulations
            climate_events: Climate event indicators
            
        Returns:
            Dictionary with Greek values
        """
        # Delta (sensitivity to underlying price)
        dS = 0.01 * S0
        price_up = self.calculate_option_price_mc(S0 + dS, K, T, r, option_type, 
                                                 n_simulations, climate_events)['option_price']
        price_down = self.calculate_option_price_mc(S0 - dS, K, T, r, option_type, 
                                                   n_simulations, climate_events)['option_price']
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma (second derivative w.r.t. underlying price)
        price_center = self.calculate_option_price_mc(S0, K, T, r, option_type, 
                                                     n_simulations, climate_events)['option_price']
        gamma = (price_up - 2 * price_center + price_down) / (dS ** 2)
        
        # Theta (sensitivity to time)
        if T > 1/365:  # At least one day
            dT = 1/365  # One day
            price_theta = self.calculate_option_price_mc(S0, K, T - dT, r, option_type, 
                                                        n_simulations, climate_events)['option_price']
            theta = (price_theta - price_center) / dT
        else:
            theta = 0.0
        
        # Rho (sensitivity to interest rate)
        dr = 0.01  # 1% change
        price_rho = self.calculate_option_price_mc(S0, K, T, r + dr, option_type, 
                                                  n_simulations, climate_events)['option_price']
        rho = (price_rho - price_center) / dr
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'rho': rho
        }
    
    def fit_parameters(self, returns: np.ndarray, 
                      climate_events: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Fit jump-diffusion parameters to observed returns.
        
        Args:
            returns: Array of observed returns
            climate_events: Array of climate event indicators
            
        Returns:
            Dictionary with fitted parameters
        """
        returns = np.asarray(returns)
        
        # Initial parameter estimates
        mu_init = np.mean(returns)
        sigma_init = np.std(returns)
        
        # Identify potential jumps (returns > 3 standard deviations)
        threshold = 3 * sigma_init
        jump_mask = np.abs(returns - mu_init) > threshold
        
        # Estimate jump parameters
        if np.any(jump_mask):
            jump_returns = returns[jump_mask]
            jump_intensity_est = np.sum(jump_mask) / len(returns)
            jump_mean_est = np.mean(jump_returns - mu_init)
            jump_std_est = np.std(jump_returns - mu_init)
        else:
            jump_intensity_est = 0.01
            jump_mean_est = 0.0
            jump_std_est = 0.01
        
        # Refine diffusion parameters (excluding jumps)
        non_jump_returns = returns[~jump_mask]
        if len(non_jump_returns) > 0:
            mu_refined = np.mean(non_jump_returns)
            sigma_refined = np.std(non_jump_returns)
        else:
            mu_refined = mu_init
            sigma_refined = sigma_init
        
        # Adjust for climate events if provided
        if climate_events is not None:
            climate_mask = climate_events > 0
            if np.any(climate_mask):
                climate_jump_mask = jump_mask & climate_mask
                if np.any(climate_jump_mask):
                    climate_jump_intensity = np.sum(climate_jump_mask) / np.sum(climate_mask)
                    jump_intensity_est = max(jump_intensity_est, climate_jump_intensity)
        
        fitted_params = {
            'mu': mu_refined,
            'sigma': sigma_refined,
            'jump_intensity': jump_intensity_est,
            'jump_mean': jump_mean_est,
            'jump_std': jump_std_est
        }
        
        # Update model parameters
        self.mu = fitted_params['mu']
        self.sigma = fitted_params['sigma']
        self.jump_intensity = fitted_params['jump_intensity']
        self.jump_mean = fitted_params['jump_mean']
        self.jump_std = fitted_params['jump_std']
        
        return fitted_params
    
    def calculate_var(self, S0: float, T: float, confidence_level: float = 0.05,
                     n_simulations: int = 100000, 
                     climate_events: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate Value at Risk using Monte Carlo simulation.
        
        Args:
            S0: Initial value
            T: Time horizon
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            n_simulations: Number of simulations
            climate_events: Climate event indicators
            
        Returns:
            Dictionary with VaR statistics
        """
        steps = max(1, int(252 * T))  # Daily steps
        
        # Simulate paths
        paths = self.simulate_multiple_paths(T, steps, n_simulations, S0, climate_events)
        final_values = paths[:, -1]
        
        # Calculate returns
        returns = (final_values - S0) / S0
        
        # Calculate VaR
        var_absolute = np.percentile(final_values - S0, confidence_level * 100)
        var_relative = np.percentile(returns, confidence_level * 100)
        
        # Calculate Expected Shortfall (Conditional VaR)
        es_mask = returns <= var_relative
        expected_shortfall = np.mean(returns[es_mask]) if np.any(es_mask) else var_relative
        
        return {
            'var_absolute': var_absolute,
            'var_relative': var_relative,
            'expected_shortfall': expected_shortfall,
            'confidence_level': confidence_level,
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns)
        }
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary with model parameters
        """
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'jump_intensity': self.jump_intensity,
            'jump_mean': self.jump_mean,
            'jump_std': self.jump_std
        }
