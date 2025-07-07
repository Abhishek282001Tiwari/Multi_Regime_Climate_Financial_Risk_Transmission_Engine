"""
Granger Causality Tests
Implements Granger causality tests for climate-financial relationships.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

class CausalityAnalyzer:
    """Granger Causality Analysis for climate-financial relationships."""
    
    def __init__(self, max_lags: int = 10):
        self.max_lags = max_lags
        self.logger = logging.getLogger(__name__)
        
    def granger_test(self, x: np.ndarray, y: np.ndarray, lags: int = 1) -> Dict[str, float]:
        """
        Perform Granger causality test.
        
        Args:
            x: Potential causal variable
            y: Target variable
            lags: Number of lags to test
            
        Returns:
            Dictionary with test results
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        if len(x) != len(y):
            raise ValueError("Variables must have the same length")
        
        n = len(x)
        if n <= lags:
            raise ValueError("Not enough observations for the specified lags")
        
        # Prepare data
        y_lagged = self._create_lags(y, lags)
        x_lagged = self._create_lags(x, lags)
        
        # Remove first 'lags' observations due to lagging
        y_target = y[lags:]
        
        # Restricted model: y regressed on its own lags
        X_restricted = y_lagged
        reg_restricted = LinearRegression()
        reg_restricted.fit(X_restricted, y_target)
        y_pred_restricted = reg_restricted.predict(X_restricted)
        rss_restricted = np.sum((y_target - y_pred_restricted) ** 2)
        
        # Unrestricted model: y regressed on its own lags and x's lags
        X_unrestricted = np.hstack([y_lagged, x_lagged])
        reg_unrestricted = LinearRegression()
        reg_unrestricted.fit(X_unrestricted, y_target)
        y_pred_unrestricted = reg_unrestricted.predict(X_unrestricted)
        rss_unrestricted = np.sum((y_target - y_pred_unrestricted) ** 2)
        
        # Calculate F-statistic
        n_obs = len(y_target)
        n_restricted = X_restricted.shape[1]
        n_unrestricted = X_unrestricted.shape[1]
        
        if rss_unrestricted == 0:
            f_stat = np.inf
        else:
            f_stat = ((rss_restricted - rss_unrestricted) / (n_unrestricted - n_restricted)) / \
                     (rss_unrestricted / (n_obs - n_unrestricted))
        
        # Calculate p-value
        df1 = n_unrestricted - n_restricted
        df2 = n_obs - n_unrestricted
        
        if df2 <= 0:
            p_value = np.nan
        else:
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        # Calculate R-squared values
        tss = np.sum((y_target - np.mean(y_target)) ** 2)
        r2_restricted = 1 - rss_restricted / tss if tss > 0 else 0
        r2_unrestricted = 1 - rss_unrestricted / tss if tss > 0 else 0
        
        return {\n            'f_statistic': f_stat,\n            'p_value': p_value,\n            'lags': lags,\n            'r2_restricted': r2_restricted,\n            'r2_unrestricted': r2_unrestricted,\n            'rss_restricted': rss_restricted,\n            'rss_unrestricted': rss_unrestricted,\n            'significant': p_value < 0.05 if not np.isnan(p_value) else False\n        }
    
    def _create_lags(self, data: np.ndarray, lags: int) -> np.ndarray:
        """Create lagged variables."""
        n = len(data)
        lagged_data = np.zeros((n - lags, lags))
        
        for i in range(lags):
            lagged_data[:, i] = data[lags - i - 1:n - i - 1]
        
        return lagged_data
    
    def test_all_pairs(self, data: pd.DataFrame, lags: int = 1) -> pd.DataFrame:
        """
        Test Granger causality for all pairs of variables.
        
        Args:
            data: DataFrame with variables
            lags: Number of lags to test
            
        Returns:
            DataFrame with test results
        """
        variables = data.columns.tolist()
        results = []
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:  # Don't test variable with itself
                    try:
                        result = self.granger_test(data[var1].values, data[var2].values, lags)
                        result['cause'] = var1
                        result['effect'] = var2
                        results.append(result)
                    except Exception as e:
                        self.logger.warning(f"Error testing {var1} -> {var2}: {e}")
                        continue
        
        return pd.DataFrame(results)
    
    def optimal_lag_selection(self, x: np.ndarray, y: np.ndarray, 
                             criteria: str = 'aic') -> Dict[str, Union[int, float]]:
        """
        Select optimal lag length using information criteria.
        
        Args:
            x: Potential causal variable
            y: Target variable
            criteria: Information criterion ('aic', 'bic', 'hqic')
            
        Returns:
            Dictionary with optimal lag and criterion value
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        best_lag = 1
        best_criterion = np.inf
        criterion_values = []
        
        for lag in range(1, min(self.max_lags + 1, len(x) // 3)):
            try:
                # Fit unrestricted model
                y_lagged = self._create_lags(y, lag)
                x_lagged = self._create_lags(x, lag)
                y_target = y[lag:]
                
                X = np.hstack([y_lagged, x_lagged])
                reg = LinearRegression()
                reg.fit(X, y_target)
                y_pred = reg.predict(X)
                
                # Calculate residual sum of squares
                rss = np.sum((y_target - y_pred) ** 2)
                n = len(y_target)
                k = X.shape[1]
                
                # Calculate information criterion
                if criteria == 'aic':
                    criterion = n * np.log(rss / n) + 2 * k
                elif criteria == 'bic':
                    criterion = n * np.log(rss / n) + k * np.log(n)
                elif criteria == 'hqic':
                    criterion = n * np.log(rss / n) + 2 * k * np.log(np.log(n))
                else:
                    raise ValueError("Criteria must be 'aic', 'bic', or 'hqic'")
                
                criterion_values.append(criterion)
                
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_lag = lag
                    
            except Exception as e:
                self.logger.warning(f"Error calculating criterion for lag {lag}: {e}")
                continue
        
        return {
            'optimal_lag': best_lag,
            'criterion_value': best_criterion,
            'all_criteria': criterion_values
        }
    
    def rolling_causality_test(self, x: np.ndarray, y: np.ndarray, 
                              window: int = 100, lags: int = 1) -> pd.DataFrame:
        """
        Perform rolling Granger causality test.
        
        Args:
            x: Potential causal variable
            y: Target variable
            window: Rolling window size
            lags: Number of lags to test
            
        Returns:
            DataFrame with rolling test results
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        if len(x) != len(y):
            raise ValueError("Variables must have the same length")
        
        n = len(x)
        if n <= window:
            raise ValueError("Not enough observations for the specified window")
        
        results = []
        
        for i in range(window, n):
            x_window = x[i - window:i]
            y_window = y[i - window:i]
            
            try:
                result = self.granger_test(x_window, y_window, lags)
                result['period'] = i
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Error in rolling test at period {i}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def test_climate_financial_causality(self, climate_data: pd.DataFrame, 
                                        financial_data: pd.DataFrame, 
                                        lags: int = 1) -> pd.DataFrame:
        """
        Test causality between climate and financial variables.
        
        Args:
            climate_data: DataFrame with climate variables
            financial_data: DataFrame with financial variables
            lags: Number of lags to test
            
        Returns:
            DataFrame with causality test results
        """
        results = []
        
        # Align data by date
        if 'date' in climate_data.columns and 'date' in financial_data.columns:
            merged_data = pd.merge(climate_data, financial_data, on='date', how='inner')
        else:
            # Assume both have datetime index
            merged_data = pd.merge(climate_data, financial_data, 
                                 left_index=True, right_index=True, how='inner')
        
        climate_vars = [col for col in merged_data.columns if col in climate_data.columns]
        financial_vars = [col for col in merged_data.columns if col in financial_data.columns]
        
        # Test climate -> financial causality
        for climate_var in climate_vars:
            for financial_var in financial_vars:
                try:
                    result = self.granger_test(
                        merged_data[climate_var].values, 
                        merged_data[financial_var].values, 
                        lags
                    )
                    result['climate_variable'] = climate_var
                    result['financial_variable'] = financial_var
                    result['direction'] = 'climate_to_financial'
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Error testing {climate_var} -> {financial_var}: {e}")
                    continue
        
        # Test financial -> climate causality
        for financial_var in financial_vars:
            for climate_var in climate_vars:
                try:
                    result = self.granger_test(
                        merged_data[financial_var].values, 
                        merged_data[climate_var].values, 
                        lags
                    )
                    result['climate_variable'] = climate_var
                    result['financial_variable'] = financial_var
                    result['direction'] = 'financial_to_climate'
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Error testing {financial_var} -> {climate_var}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def create_causality_network(self, causality_results: pd.DataFrame, 
                               significance_level: float = 0.05) -> Dict[str, List[str]]:
        """
        Create causality network from test results.
        
        Args:
            causality_results: DataFrame with causality test results
            significance_level: Significance level for filtering
            
        Returns:
            Dictionary representing causality network
        """
        # Filter significant results
        significant_results = causality_results[
            causality_results['p_value'] < significance_level
        ]
        
        network = {}
        
        for _, row in significant_results.iterrows():
            if 'cause' in row and 'effect' in row:
                cause = row['cause']
                effect = row['effect']
            elif 'climate_variable' in row and 'financial_variable' in row:
                if row['direction'] == 'climate_to_financial':
                    cause = row['climate_variable']
                    effect = row['financial_variable']
                else:
                    cause = row['financial_variable']
                    effect = row['climate_variable']
            else:
                continue
            
            if cause not in network:
                network[cause] = []
            network[cause].append(effect)
        
        return network
    
    def summarize_causality_results(self, causality_results: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """
        Summarize causality test results.
        
        Args:
            causality_results: DataFrame with causality test results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_tests': len(causality_results),
            'significant_at_5pct': len(causality_results[causality_results['p_value'] < 0.05]),
            'significant_at_1pct': len(causality_results[causality_results['p_value'] < 0.01]),
            'mean_f_statistic': causality_results['f_statistic'].mean(),
            'mean_p_value': causality_results['p_value'].mean(),
            'proportion_significant_5pct': len(causality_results[causality_results['p_value'] < 0.05]) / len(causality_results),
            'proportion_significant_1pct': len(causality_results[causality_results['p_value'] < 0.01]) / len(causality_results)
        }
        
        return summary
