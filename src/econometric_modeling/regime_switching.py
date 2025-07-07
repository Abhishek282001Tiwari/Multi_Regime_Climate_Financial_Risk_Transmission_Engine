"""
Markov Regime-Switching Models
Implements various regime-switching models for financial time series.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class MarkovRegimeSwitching:
    """Markov Regime-Switching Model for financial time series."""
    
    def __init__(self, n_regimes: int = 2, max_iter: int = 1000, tol: float = 1e-6):
        self.n_regimes = n_regimes
        self.max_iter = max_iter
        self.tol = tol
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.transition_matrix = None
        self.regime_params = None
        self.regime_probabilities = None
        self.log_likelihood = None
        self.fitted = False
        
    def fit(self, data: np.ndarray) -> 'MarkovRegimeSwitching':
        """
        Fit the Markov regime-switching model.
        
        Args:
            data: Time series data
            
        Returns:
            Fitted model
        """
        data = np.asarray(data).flatten()
        n_obs = len(data)
        
        # Initialize parameters using Gaussian Mixture Model
        self._initialize_parameters(data)
        
        # EM algorithm
        log_likelihood_old = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: Calculate regime probabilities
            self._expectation_step(data)
            
            # M-step: Update parameters
            self._maximization_step(data)
            
            # Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood(data)
            
            # Check convergence
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break
                
            log_likelihood_old = log_likelihood
        
        self.log_likelihood = log_likelihood
        self.fitted = True
        
        return self
    
    def _initialize_parameters(self, data: np.ndarray):
        """Initialize model parameters using Gaussian Mixture Model."""
        # Use GMM for initial parameter estimation
        gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
        gmm.fit(data.reshape(-1, 1))
        
        # Initialize regime parameters
        self.regime_params = {}
        for i in range(self.n_regimes):
            self.regime_params[i] = {
                'mean': gmm.means_[i, 0],
                'variance': gmm.covariances_[i, 0, 0]
            }
        
        # Initialize transition matrix (uniform)
        self.transition_matrix = np.full((self.n_regimes, self.n_regimes), 
                                       1.0 / self.n_regimes)
        
        # Initialize regime probabilities
        n_obs = len(data)
        self.regime_probabilities = np.full((n_obs, self.n_regimes), 
                                          1.0 / self.n_regimes)
    
    def _expectation_step(self, data: np.ndarray):
        """E-step: Calculate regime probabilities using forward-backward algorithm."""
        n_obs = len(data)
        
        # Forward probabilities
        forward_probs = np.zeros((n_obs, self.n_regimes))
        
        # Initial probabilities
        for i in range(self.n_regimes):
            forward_probs[0, i] = self._emission_probability(data[0], i)
        
        # Normalize
        forward_probs[0] /= np.sum(forward_probs[0])
        
        # Forward recursion
        for t in range(1, n_obs):
            for i in range(self.n_regimes):
                forward_probs[t, i] = self._emission_probability(data[t], i) * \
                                    np.sum(forward_probs[t-1] * self.transition_matrix[:, i])
            
            # Normalize
            forward_probs[t] /= np.sum(forward_probs[t])
        
        # Backward probabilities
        backward_probs = np.zeros((n_obs, self.n_regimes))
        backward_probs[-1] = 1.0
        
        # Backward recursion
        for t in range(n_obs - 2, -1, -1):
            for i in range(self.n_regimes):
                backward_probs[t, i] = np.sum(
                    self.transition_matrix[i] * 
                    backward_probs[t+1] * 
                    np.array([self._emission_probability(data[t+1], j) 
                             for j in range(self.n_regimes)])
                )
            
            # Normalize
            if np.sum(backward_probs[t]) > 0:
                backward_probs[t] /= np.sum(backward_probs[t])
        
        # Combine forward and backward probabilities
        self.regime_probabilities = forward_probs * backward_probs
        
        # Normalize
        for t in range(n_obs):
            if np.sum(self.regime_probabilities[t]) > 0:
                self.regime_probabilities[t] /= np.sum(self.regime_probabilities[t])
    
    def _maximization_step(self, data: np.ndarray):
        """M-step: Update model parameters."""
        n_obs = len(data)
        
        # Update regime parameters
        for i in range(self.n_regimes):
            weights = self.regime_probabilities[:, i]
            weight_sum = np.sum(weights)
            
            if weight_sum > 0:
                # Update mean
                self.regime_params[i]['mean'] = np.sum(weights * data) / weight_sum
                
                # Update variance
                squared_deviations = (data - self.regime_params[i]['mean']) ** 2
                self.regime_params[i]['variance'] = np.sum(weights * squared_deviations) / weight_sum
                
                # Ensure positive variance
                self.regime_params[i]['variance'] = max(self.regime_params[i]['variance'], 1e-6)
        
        # Update transition matrix
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                numerator = np.sum(self.regime_probabilities[:-1, i] * self.regime_probabilities[1:, j])
                denominator = np.sum(self.regime_probabilities[:-1, i])
                
                if denominator > 0:
                    self.transition_matrix[i, j] = numerator / denominator
                else:
                    self.transition_matrix[i, j] = 1.0 / self.n_regimes
        
        # Normalize transition matrix rows
        for i in range(self.n_regimes):
            row_sum = np.sum(self.transition_matrix[i])
            if row_sum > 0:
                self.transition_matrix[i] /= row_sum
    
    def _emission_probability(self, observation: float, regime: int) -> float:
        """Calculate emission probability for observation in given regime."""
        mean = self.regime_params[regime]['mean']
        variance = self.regime_params[regime]['variance']
        
        return norm.pdf(observation, loc=mean, scale=np.sqrt(variance))
    
    def _calculate_log_likelihood(self, data: np.ndarray) -> float:
        """Calculate log-likelihood of the model."""
        log_likelihood = 0.0
        
        for t, observation in enumerate(data):
            prob_sum = 0.0
            for i in range(self.n_regimes):
                prob_sum += self.regime_probabilities[t, i] * \
                           self._emission_probability(observation, i)
            
            if prob_sum > 0:
                log_likelihood += np.log(prob_sum)
        
        return log_likelihood
    
    def predict_regime(self, data: np.ndarray) -> np.ndarray:
        """
        Predict most likely regime for each observation.
        
        Args:
            data: Time series data
            
        Returns:
            Array of predicted regimes
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        data = np.asarray(data).flatten()
        n_obs = len(data)
        
        # Calculate regime probabilities
        self._expectation_step(data)
        
        # Return most likely regime
        return np.argmax(self.regime_probabilities, axis=1)
    
    def regime_probabilities_over_time(self, data: np.ndarray) -> pd.DataFrame:
        """
        Get regime probabilities over time.
        
        Args:
            data: Time series data
            
        Returns:
            DataFrame with regime probabilities
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        data = np.asarray(data).flatten()
        self._expectation_step(data)
        
        columns = [f'Regime_{i}' for i in range(self.n_regimes)]
        return pd.DataFrame(self.regime_probabilities, columns=columns)
    
    def get_regime_statistics(self) -> Dict:
        """
        Get statistics for each regime.
        
        Returns:
            Dictionary with regime statistics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting statistics")
        
        stats = {}
        for i in range(self.n_regimes):
            stats[f'Regime_{i}'] = {
                'mean': self.regime_params[i]['mean'],
                'variance': self.regime_params[i]['variance'],
                'std': np.sqrt(self.regime_params[i]['variance'])
            }
        
        return stats
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get transition matrix as DataFrame.
        
        Returns:
            Transition matrix DataFrame
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting transition matrix")
        
        index = [f'Regime_{i}' for i in range(self.n_regimes)]
        columns = [f'Regime_{i}' for i in range(self.n_regimes)]
        
        return pd.DataFrame(self.transition_matrix, index=index, columns=columns)
    
    def forecast_regime(self, steps: int, current_regime: int = None) -> np.ndarray:
        """
        Forecast regime probabilities for future steps.
        
        Args:
            steps: Number of steps to forecast
            current_regime: Current regime (if None, use steady state)
            
        Returns:
            Array of forecasted regime probabilities
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        if current_regime is None:
            # Use steady state probabilities
            eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
            stationary_idx = np.argmin(np.abs(eigenvals - 1.0))
            current_probs = np.real(eigenvecs[:, stationary_idx])
            current_probs = current_probs / np.sum(current_probs)
        else:
            current_probs = np.zeros(self.n_regimes)
            current_probs[current_regime] = 1.0
        
        # Forecast using transition matrix
        forecasts = np.zeros((steps, self.n_regimes))
        probs = current_probs.copy()
        
        for t in range(steps):
            probs = probs @ self.transition_matrix
            forecasts[t] = probs.copy()
        
        return forecasts
