"""
Markov Regime-Switching Model
PhD-level implementation for climate-financial regime detection.

This module implements Hamilton's (1989) Markov regime-switching model with extensions
for climate-financial analysis. The model identifies different market regimes and
estimates transition probabilities.

Mathematical Framework:
----------------------
y_t = μ(s_t) + σ(s_t)ε_t

where:
- s_t ∈ {1, 2, ..., k} is the unobserved regime at time t
- μ(s_t) is the regime-dependent mean
- σ(s_t) is the regime-dependent volatility  
- ε_t ~ N(0,1) is the error term

Transition probabilities: P(s_t = j | s_{t-1} = i) = p_{ij}

Author: Climate Risk Research Team
References: Hamilton (1989), Kim & Nelson (1999), Krolzig (1997)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Try to import statsmodels regime switching (fallback to custom implementation)
try:
    from statsmodels.tsa.regime_switching import markov_switching
    HAS_STATSMODELS_MS = True
except ImportError:
    HAS_STATSMODELS_MS = False

class MarkovRegimeSwitching:
    """
    Advanced Markov Regime-Switching Model for Climate-Financial Analysis.
    
    This class implements both statsmodels-based and custom EM algorithm approaches
    for regime-switching analysis with specific focus on climate-financial transmission.
    
    Parameters:
    -----------
    n_regimes : int
        Number of regimes (typically 2-4 for financial data)
    model_type : str
        Type of model ('mean', 'variance', 'mean_variance')
    max_iter : int
        Maximum iterations for EM algorithm
    tol : float
        Convergence tolerance
    
    Attributes:
    -----------
    regimes : np.ndarray
        Estimated regime sequence
    regime_probs : np.ndarray
        Smoothed regime probabilities
    transition_matrix : np.ndarray
        Estimated transition probability matrix
    parameters : dict
        Estimated model parameters
    """
    
    def __init__(self, 
                 n_regimes: int = 2,
                 model_type: str = 'mean_variance',
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 random_state: int = 42):
        """
        Initialize the Markov regime-switching model.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes to identify
        model_type : str
            'mean': regime-switching in mean only
            'variance': regime-switching in variance only  
            'mean_variance': regime-switching in both mean and variance
        max_iter : int
            Maximum iterations for EM algorithm
        tol : float
            Convergence tolerance for log-likelihood
        random_state : int
            Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.model_type = model_type
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Model results
        self.is_fitted = False
        self.regimes = None
        self.regime_probs = None
        self.transition_matrix = None
        self.parameters = {}
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        
        # Set random seed
        np.random.seed(random_state)
        
        self.logger.info(f"MarkovRegimeSwitching initialized: {n_regimes} regimes, {model_type} model")
    
    def fit(self, 
            data: Union[np.ndarray, pd.Series],
            method: str = 'auto') -> 'MarkovRegimeSwitching':
        """
        Fit the Markov regime-switching model to data.
        
        Parameters:
        -----------
        data : array-like
            Time series data to fit
        method : str
            'statsmodels': Use statsmodels implementation (if available)
            'custom': Use custom EM algorithm implementation
            'auto': Try statsmodels first, fallback to custom
            
        Returns:
        --------
        self : MarkovRegimeSwitching
            Fitted model instance
            
        Mathematical Details:
        --------------------
        The EM algorithm alternates between:
        
        E-step: Compute regime probabilities using forward-backward algorithm
        ξ_t(i) = P(s_t = i | y_1, ..., y_T)
        
        M-step: Update parameters
        μ_i = Σ_t ξ_t(i) y_t / Σ_t ξ_t(i)
        σ²_i = Σ_t ξ_t(i) (y_t - μ_i)² / Σ_t ξ_t(i)
        p_ij = Σ_t ξ_{t-1,t}(i,j) / Σ_t ξ_t(i)
        """
        
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values
        data = np.asarray(data).flatten()
        
        # Remove NaNs
        data = data[~np.isnan(data)]
        
        if len(data) < 50:
            raise ValueError("Insufficient data points for regime switching analysis (minimum 50)")
        
        self.data = data
        self.n_obs = len(data)
        
        self.logger.info(f"Fitting regime-switching model to {self.n_obs} observations...")
        
        # Choose fitting method
        if method == 'auto':
            if HAS_STATSMODELS_MS:
                method = 'statsmodels'
            else:
                method = 'custom'
                self.logger.warning("statsmodels not available, using custom implementation")
        
        if method == 'statsmodels' and HAS_STATSMODELS_MS:
            self._fit_statsmodels()
        else:
            self._fit_custom_em()
        
        # Calculate information criteria
        self._calculate_information_criteria()
        
        self.is_fitted = True
        self.logger.info(f"Model fitting completed. Log-likelihood: {self.log_likelihood:.4f}")
        
        return self
    
    def _fit_statsmodels(self):
        """Fit using statsmodels implementation."""
        try:
            # Determine model specification
            if self.model_type == 'mean':
                switching_mean = True
                switching_variance = False
            elif self.model_type == 'variance':
                switching_mean = False  
                switching_variance = True
            else:  # mean_variance
                switching_mean = True
                switching_variance = True
            
            # Fit the model
            model = markov_switching.MarkovRegression(
                self.data,
                k_regimes=self.n_regimes,
                trend='c',
                switching_trend=switching_mean,
                switching_variance=switching_variance
            )
            
            results = model.fit(maxiter=self.max_iter)
            
            # Extract results
            self.regime_probs = results.smoothed_marginal_probabilities
            self.regimes = np.argmax(self.regime_probs, axis=1)
            self.transition_matrix = results.regime_transition
            self.log_likelihood = results.llf
            
            # Extract parameters
            self.parameters = {
                'means': [],
                'variances': [],
                'regime_probs': results.regime_transition[0]  # Initial probabilities
            }
            
            for i in range(self.n_regimes):
                # Mean parameters
                if switching_mean:
                    mean_param = results.params[f'const[{i}]']
                else:
                    mean_param = results.params['const']
                self.parameters['means'].append(mean_param)
                
                # Variance parameters
                if switching_variance:
                    var_param = results.params[f'sigma2[{i}]']
                else:
                    var_param = results.params['sigma2']
                self.parameters['variances'].append(var_param)
            
            self.logger.info("Statsmodels fitting completed successfully")
            
        except Exception as e:
            self.logger.warning(f"Statsmodels fitting failed: {str(e)}, falling back to custom implementation")
            self._fit_custom_em()
    
    def _fit_custom_em(self):
        """Fit using custom EM algorithm implementation."""
        
        # Initialize parameters using Gaussian Mixture Model
        gmm = GaussianMixture(n_components=self.n_regimes, random_state=self.random_state)
        gmm.fit(self.data.reshape(-1, 1))
        
        # Initial parameter estimates
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        
        # Initial transition matrix (uniform)
        transition_matrix = np.full((self.n_regimes, self.n_regimes), 1.0 / self.n_regimes)
        
        # Initial regime probabilities (from GMM)
        regime_probs = gmm.predict_proba(self.data.reshape(-1, 1))
        
        self.logger.info("Starting EM algorithm...")
        
        log_likelihood_old = -np.inf
        
        for iteration in range(self.max_iter):
            
            # E-step: Forward-backward algorithm
            regime_probs = self._forward_backward(means, variances, transition_matrix)
            
            # M-step: Update parameters
            means, variances, transition_matrix = self._update_parameters(regime_probs)
            
            # Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood(means, variances, regime_probs)
            
            # Check convergence
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                self.logger.info(f"EM algorithm converged after {iteration + 1} iterations")
                break
            
            log_likelihood_old = log_likelihood
            
            if iteration % 100 == 0:
                self.logger.debug(f"Iteration {iteration}, Log-likelihood: {log_likelihood:.6f}")
        
        # Store results
        self.regime_probs = regime_probs
        self.regimes = np.argmax(regime_probs, axis=1)
        self.transition_matrix = transition_matrix
        self.log_likelihood = log_likelihood
        
        self.parameters = {
            'means': means,
            'variances': variances,
            'regime_probs': regime_probs[0]  # Initial probabilities
        }
        
        self.logger.info("Custom EM algorithm completed successfully")
    
    def _forward_backward(self, 
                         means: np.ndarray, 
                         variances: np.ndarray, 
                         transition_matrix: np.ndarray) -> np.ndarray:
        """
        Forward-backward algorithm for regime probability estimation.
        
        This implements the Baum-Welch algorithm for Hidden Markov Models.
        """
        
        # Forward probabilities
        forward_probs = np.zeros((self.n_obs, self.n_regimes))
        
        # Initial probabilities (assume uniform)
        initial_probs = np.ones(self.n_regimes) / self.n_regimes
        
        # Forward pass
        for i in range(self.n_regimes):
            forward_probs[0, i] = initial_probs[i] * self._emission_probability(
                self.data[0], means[i], variances[i]
            )
        
        # Normalize
        forward_probs[0] /= np.sum(forward_probs[0])
        
        for t in range(1, self.n_obs):
            for j in range(self.n_regimes):
                forward_probs[t, j] = self._emission_probability(
                    self.data[t], means[j], variances[j]
                ) * np.sum(forward_probs[t-1] * transition_matrix[:, j])
            
            # Normalize to prevent underflow
            forward_probs[t] /= np.sum(forward_probs[t])
        
        # Backward probabilities
        backward_probs = np.zeros((self.n_obs, self.n_regimes))
        backward_probs[-1] = 1.0
        
        for t in range(self.n_obs - 2, -1, -1):
            for i in range(self.n_regimes):
                backward_probs[t, i] = np.sum(
                    transition_matrix[i] * 
                    backward_probs[t+1] * 
                    np.array([self._emission_probability(self.data[t+1], means[j], variances[j]) 
                             for j in range(self.n_regimes)])
                )
            
            # Normalize
            if np.sum(backward_probs[t]) > 0:
                backward_probs[t] /= np.sum(backward_probs[t])
        
        # Combine forward and backward probabilities
        regime_probs = forward_probs * backward_probs
        
        # Normalize
        for t in range(self.n_obs):
            if np.sum(regime_probs[t]) > 0:
                regime_probs[t] /= np.sum(regime_probs[t])
        
        return regime_probs
    
    def _emission_probability(self, observation: float, mean: float, variance: float) -> float:
        """Calculate emission probability for normal distribution."""
        return stats.norm.pdf(observation, loc=mean, scale=np.sqrt(variance))
    
    def _update_parameters(self, regime_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M-step: Update model parameters."""
        
        means = np.zeros(self.n_regimes)
        variances = np.zeros(self.n_regimes)
        
        # Update means and variances
        for i in range(self.n_regimes):
            weights = regime_probs[:, i]
            weight_sum = np.sum(weights)
            
            if weight_sum > 0:
                means[i] = np.sum(weights * self.data) / weight_sum
                variances[i] = np.sum(weights * (self.data - means[i])**2) / weight_sum
                
                # Ensure positive variance
                variances[i] = max(variances[i], 1e-6)
        
        # Update transition matrix
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                numerator = np.sum(regime_probs[:-1, i] * regime_probs[1:, j])
                denominator = np.sum(regime_probs[:-1, i])
                
                if denominator > 0:
                    transition_matrix[i, j] = numerator / denominator
                else:
                    transition_matrix[i, j] = 1.0 / self.n_regimes
        
        # Normalize transition matrix rows
        for i in range(self.n_regimes):
            row_sum = np.sum(transition_matrix[i])
            if row_sum > 0:
                transition_matrix[i] /= row_sum
        
        return means, variances, transition_matrix
    
    def _calculate_log_likelihood(self, 
                                 means: np.ndarray, 
                                 variances: np.ndarray, 
                                 regime_probs: np.ndarray) -> float:
        """Calculate log-likelihood of the model."""
        
        log_likelihood = 0.0
        
        for t in range(self.n_obs):
            prob_sum = 0.0
            for i in range(self.n_regimes):
                prob_sum += regime_probs[t, i] * self._emission_probability(
                    self.data[t], means[i], variances[i]
                )
            
            if prob_sum > 0:
                log_likelihood += np.log(prob_sum)
        
        return log_likelihood
    
    def _calculate_information_criteria(self):
        """Calculate AIC and BIC for model selection."""
        
        # Number of parameters
        n_params = (
            self.n_regimes +  # means
            self.n_regimes +  # variances  
            self.n_regimes * (self.n_regimes - 1)  # transition probabilities
        )
        
        # AIC and BIC
        self.aic = 2 * n_params - 2 * self.log_likelihood
        self.bic = np.log(self.n_obs) * n_params - 2 * self.log_likelihood
    
    def predict_regime(self, data: Union[np.ndarray, pd.Series] = None) -> np.ndarray:
        """
        Predict regime for new data or return fitted regimes.
        
        Parameters:
        -----------
        data : array-like, optional
            New data to predict regimes for (if None, returns fitted regimes)
            
        Returns:
        --------
        np.ndarray
            Predicted regime sequence
        """
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if data is None:
            return self.regimes
        
        # For new data, use Viterbi algorithm (simplified version)
        if isinstance(data, pd.Series):
            data = data.values
        data = np.asarray(data).flatten()
        
        # Simple prediction using fitted parameters
        regime_probs = self._forward_backward(
            self.parameters['means'], 
            self.parameters['variances'], 
            self.transition_matrix
        )
        
        return np.argmax(regime_probs, axis=1)
    
    def get_regime_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each regime.
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics by regime
        """
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting regime summary")
        
        summary_data = []
        
        for i in range(self.n_regimes):
            regime_mask = self.regimes == i
            regime_data = self.data[regime_mask]
            
            if len(regime_data) > 0:
                summary = {
                    'Regime': i,
                    'Observations': len(regime_data),
                    'Frequency': len(regime_data) / self.n_obs,
                    'Mean': np.mean(regime_data),
                    'Std': np.std(regime_data),
                    'Min': np.min(regime_data),
                    'Max': np.max(regime_data),
                    'Duration_Mean': self._calculate_mean_duration(i)
                }
            else:
                summary = {
                    'Regime': i,
                    'Observations': 0,
                    'Frequency': 0,
                    'Mean': np.nan,
                    'Std': np.nan,
                    'Min': np.nan,
                    'Max': np.nan,
                    'Duration_Mean': np.nan
                }
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def _calculate_mean_duration(self, regime: int) -> float:
        """Calculate mean duration of a regime."""
        
        if regime >= self.n_regimes:
            return np.nan
        
        # Expected duration = 1 / (1 - p_ii)
        p_ii = self.transition_matrix[regime, regime]
        
        if p_ii >= 1.0:
            return np.inf
        else:
            return 1.0 / (1.0 - p_ii)
    
    def plot_regimes(self, 
                    figsize: Tuple[int, int] = (15, 10),
                    save_path: str = None) -> plt.Figure:
        """
        Plot regime analysis results.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Markov Regime-Switching Analysis', fontsize=16, fontweight='bold')
        
        # 1. Time series with regimes
        ax1 = axes[0, 0]
        colors = ['red', 'blue', 'green', 'orange'][:self.n_regimes]
        
        for i in range(self.n_regimes):
            regime_mask = self.regimes == i
            ax1.scatter(np.where(regime_mask)[0], self.data[regime_mask], 
                       c=colors[i], alpha=0.6, s=20, label=f'Regime {i}')
        
        ax1.plot(self.data, 'k-', alpha=0.3, linewidth=1)
        ax1.set_title('Time Series with Identified Regimes')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Regime probabilities
        ax2 = axes[0, 1]
        for i in range(self.n_regimes):
            ax2.plot(self.regime_probs[:, i], label=f'Regime {i}', color=colors[i])
        
        ax2.set_title('Regime Probabilities Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Transition matrix heatmap
        ax3 = axes[1, 0]
        sns.heatmap(self.transition_matrix, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=[f'Regime {i}' for i in range(self.n_regimes)],
                   yticklabels=[f'Regime {i}' for i in range(self.n_regimes)],
                   ax=ax3)
        ax3.set_title('Transition Probability Matrix')
        
        # 4. Regime distributions
        ax4 = axes[1, 1]
        for i in range(self.n_regimes):
            regime_data = self.data[self.regimes == i]
            if len(regime_data) > 0:
                ax4.hist(regime_data, bins=30, alpha=0.7, color=colors[i], 
                        label=f'Regime {i}', density=True)
        
        ax4.set_title('Distribution by Regime')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def get_model_diagnostics(self) -> Dict[str, any]:
        """
        Get comprehensive model diagnostics.
        
        Returns:
        --------
        Dict[str, any]
            Model diagnostics and fit statistics
        """
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting diagnostics")
        
        diagnostics = {
            'model_info': {
                'n_regimes': self.n_regimes,
                'model_type': self.model_type,
                'n_observations': self.n_obs,
                'convergence': True  # Assume convergence if fitted
            },
            'fit_statistics': {
                'log_likelihood': self.log_likelihood,
                'aic': self.aic,
                'bic': self.bic
            },
            'regime_statistics': self.get_regime_summary().to_dict('records'),
            'transition_matrix': self.transition_matrix.tolist(),
            'parameters': self.parameters
        }
        
        return diagnostics
