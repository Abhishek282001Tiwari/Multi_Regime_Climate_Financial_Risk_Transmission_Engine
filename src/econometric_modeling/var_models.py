"""
Vector Autoregression (VAR) Models
Implements VAR models for cross-asset contagion analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

class VARModel:
    """Vector Autoregression Model for multivariate time series."""
    
    def __init__(self, lags: int = 1, include_constant: bool = True):
        self.lags = lags
        self.include_constant = include_constant
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.coefficients = None
        self.fitted_values = None
        self.residuals = None
        self.sigma = None  # Covariance matrix of residuals
        self.variable_names = None
        self.n_variables = None
        self.n_obs = None
        self.fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'VARModel':
        """
        Fit the VAR model.
        
        Args:
            data: Multivariate time series data
            
        Returns:
            Fitted model
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        self.variable_names = data.columns.tolist()
        self.n_variables = len(self.variable_names)
        
        # Create lagged variables
        X, y = self._create_lagged_data(data)
        
        if X.shape[0] == 0:
            raise ValueError("Not enough observations for the specified number of lags")
        
        self.n_obs = X.shape[0]
        
        # Fit regression for each variable
        self.coefficients = {}
        self.fitted_values = np.zeros((self.n_obs, self.n_variables))
        self.residuals = np.zeros((self.n_obs, self.n_variables))
        
        for i, var_name in enumerate(self.variable_names):
            # Fit linear regression
            reg = LinearRegression(fit_intercept=False)  # Constant handled in X
            reg.fit(X, y[:, i])
            
            self.coefficients[var_name] = reg.coef_
            self.fitted_values[:, i] = reg.predict(X)
            self.residuals[:, i] = y[:, i] - self.fitted_values[:, i]
        
        # Calculate residual covariance matrix
        self.sigma = np.cov(self.residuals, rowvar=False)
        
        self.fitted = True
        return self
    
    def _create_lagged_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged variables for VAR model."""
        n_obs, n_vars = data.shape
        
        # Create lagged matrix
        X_list = []
        
        # Add constant if specified
        if self.include_constant:
            X_list.append(np.ones((n_obs - self.lags, 1)))
        
        # Add lagged variables
        for lag in range(1, self.lags + 1):
            lagged_data = data.shift(lag).iloc[self.lags:].values
            X_list.append(lagged_data)
        
        X = np.hstack(X_list)
        y = data.iloc[self.lags:].values
        
        return X, y
    
    def predict(self, steps: int, last_obs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Forecast future values.
        
        Args:
            steps: Number of steps to forecast
            last_obs: Last observations for prediction (if None, use training data)
            
        Returns:
            DataFrame with forecasts
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if last_obs is None:
            # Use last observations from training data
            last_obs = pd.DataFrame(self.fitted_values[-self.lags:], 
                                   columns=self.variable_names)
        
        forecasts = []
        current_obs = last_obs.copy()
        
        for step in range(steps):
            # Create input for next prediction
            X_pred = []
            
            # Add constant if specified
            if self.include_constant:
                X_pred.append([1.0])
            
            # Add lagged variables
            for lag in range(1, self.lags + 1):
                if lag <= len(current_obs):
                    X_pred.append(current_obs.iloc[-lag].values)
                else:
                    X_pred.append(np.zeros(self.n_variables))
            
            X_pred = np.hstack(X_pred)
            
            # Predict next values
            next_values = {}
            for var_name in self.variable_names:
                next_values[var_name] = np.dot(X_pred, self.coefficients[var_name])
            
            # Store forecast
            forecasts.append(next_values)
            
            # Update current observations
            new_obs = pd.DataFrame([next_values])
            current_obs = pd.concat([current_obs, new_obs], ignore_index=True)
        
        return pd.DataFrame(forecasts)
    
    def impulse_response(self, steps: int, shock_size: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Calculate impulse response functions.
        
        Args:
            steps: Number of steps for impulse response
            shock_size: Size of the shock
            
        Returns:
            Dictionary with impulse response functions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating impulse responses")
        
        # Get coefficient matrices
        coeff_matrices = self._get_coefficient_matrices()
        
        # Calculate impulse responses
        impulse_responses = {}
        
        for shock_var in self.variable_names:
            responses = np.zeros((steps, self.n_variables))
            
            # Initial shock
            shock = np.zeros(self.n_variables)
            shock_idx = self.variable_names.index(shock_var)
            shock[shock_idx] = shock_size
            
            # Store responses
            responses[0] = shock
            
            # Calculate responses for subsequent periods
            for t in range(1, steps):
                response = np.zeros(self.n_variables)
                
                for lag in range(min(t, self.lags)):
                    if lag < len(coeff_matrices):
                        response += coeff_matrices[lag] @ responses[t - lag - 1]
                
                responses[t] = response
            
            # Store as DataFrame
            impulse_responses[shock_var] = pd.DataFrame(
                responses, 
                columns=self.variable_names
            )
        
        return impulse_responses
    
    def _get_coefficient_matrices(self) -> List[np.ndarray]:
        """Get coefficient matrices for each lag."""
        matrices = []
        
        for lag in range(self.lags):
            matrix = np.zeros((self.n_variables, self.n_variables))
            
            for i, var_name in enumerate(self.variable_names):
                coef = self.coefficients[var_name]
                
                # Skip constant term
                start_idx = 1 if self.include_constant else 0
                
                # Extract coefficients for this lag
                lag_start = start_idx + lag * self.n_variables
                lag_end = lag_start + self.n_variables
                
                if lag_end <= len(coef):
                    matrix[i, :] = coef[lag_start:lag_end]
            
            matrices.append(matrix)
        
        return matrices
    
    def forecast_error_variance_decomposition(self, steps: int) -> Dict[str, pd.DataFrame]:
        """
        Calculate forecast error variance decomposition.
        
        Args:
            steps: Number of steps for FEVD
            
        Returns:
            Dictionary with variance decompositions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating FEVD")
        
        # Get impulse responses
        impulse_responses = self.impulse_response(steps)
        
        # Calculate cumulative squared responses
        fevd_results = {}
        
        for var_name in self.variable_names:
            fevd = np.zeros((steps, self.n_variables))
            
            for t in range(steps):
                # Calculate total variance up to time t
                total_variance = 0
                variance_by_shock = np.zeros(self.n_variables)
                
                for shock_idx, shock_var in enumerate(self.variable_names):
                    shock_responses = impulse_responses[shock_var][var_name].values[:t+1]
                    variance_contribution = np.sum(shock_responses ** 2)
                    variance_by_shock[shock_idx] = variance_contribution
                    total_variance += variance_contribution
                
                # Calculate proportions
                if total_variance > 0:
                    fevd[t] = variance_by_shock / total_variance
                else:
                    fevd[t] = np.zeros(self.n_variables)
            
            fevd_results[var_name] = pd.DataFrame(
                fevd, 
                columns=self.variable_names
            )
        
        return fevd_results
    
    def get_residuals(self) -> pd.DataFrame:
        """
        Get model residuals.
        
        Returns:
            DataFrame with residuals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting residuals")
        
        return pd.DataFrame(self.residuals, columns=self.variable_names)
    
    def get_fitted_values(self) -> pd.DataFrame:
        """
        Get fitted values.
        
        Returns:
            DataFrame with fitted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting fitted values")
        
        return pd.DataFrame(self.fitted_values, columns=self.variable_names)
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients.
        
        Returns:
            DataFrame with coefficients
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        
        coeff_data = []
        for var_name in self.variable_names:
            coeff_data.append(self.coefficients[var_name])
        
        return pd.DataFrame(coeff_data, index=self.variable_names)
    
    def calculate_aic(self) -> float:
        """
        Calculate Akaike Information Criterion.
        
        Returns:
            AIC value
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating AIC")
        
        # Number of parameters
        n_params = self.n_variables * (self.lags * self.n_variables + 
                                      (1 if self.include_constant else 0))
        
        # Log-likelihood
        log_likelihood = self._calculate_log_likelihood()
        
        # AIC
        aic = 2 * n_params - 2 * log_likelihood
        
        return aic
    
    def calculate_bic(self) -> float:
        """
        Calculate Bayesian Information Criterion.
        
        Returns:
            BIC value
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating BIC")
        
        # Number of parameters
        n_params = self.n_variables * (self.lags * self.n_variables + 
                                      (1 if self.include_constant else 0))
        
        # Log-likelihood
        log_likelihood = self._calculate_log_likelihood()
        
        # BIC
        bic = np.log(self.n_obs) * n_params - 2 * log_likelihood
        
        return bic
    
    def _calculate_log_likelihood(self) -> float:
        """Calculate log-likelihood of the model."""
        # Log-likelihood for multivariate normal
        log_likelihood = 0
        
        for t in range(self.n_obs):
            residual = self.residuals[t, :]
            
            # Calculate log-likelihood
            try:
                ll = -0.5 * (
                    self.n_variables * np.log(2 * np.pi) +
                    np.log(np.linalg.det(self.sigma)) +
                    residual.T @ np.linalg.inv(self.sigma) @ residual
                )
                log_likelihood += ll
            except np.linalg.LinAlgError:
                # Handle singular matrix
                log_likelihood += -1e10
        
        return log_likelihood
