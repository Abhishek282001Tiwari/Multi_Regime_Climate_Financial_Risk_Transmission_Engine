"""
Jump-Diffusion Model (Merton Model)
PhD-level implementation for climate-triggered financial jump processes.

This module implements Merton's (1976) jump-diffusion model with extensions for
climate-financial analysis. The model captures sudden price movements triggered
by climate events and other extreme market conditions.

Mathematical Framework:
----------------------
dS_t = μS_t dt + σS_t dW_t + S_t ∫_{-∞}^{∞} (e^y - 1) Ñ(dt, dy)

where:
- S_t is the asset price at time t
- μ is the drift parameter
- σ is the diffusion volatility
- W_t is a Brownian motion
- Ñ(dt, dy) is a compensated Poisson random measure
- Jump sizes Y ~ N(μ_J, σ_J²)
- Jump intensity λ (potentially climate-dependent)

For climate applications, λ = λ₀ + β × Climate_Index_t

Author: Climate Risk Research Team
References: Merton (1976), Kou (2002), Cont & Tankov (2004)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class JumpDiffusionModel:
    """
    Advanced Jump-Diffusion Model for Climate-Financial Analysis.
    
    This class implements the Merton jump-diffusion model with climate-dependent
    jump intensity, allowing for the analysis of how climate events trigger
    sudden movements in financial markets.
    
    Parameters:
    -----------
    mu : float
        Drift parameter (annual)
    sigma : float
        Diffusion volatility (annual)
    lambda_jump : float
        Base jump intensity (jumps per year)
    mu_jump : float
        Mean jump size (log scale)
    sigma_jump : float
        Jump size volatility (log scale)
    climate_beta : float
        Sensitivity of jump intensity to climate factors
    
    Mathematical Properties:
    -----------------------
    - Log returns: r_t = ln(S_t/S_{t-1}) = (μ - σ²/2 - λκ)dt + σdW_t + J_t
    - κ = E[e^Y - 1] = exp(μ_J + σ_J²/2) - 1 (mean jump size)
    - Variance: Var[r_t] = σ²dt + λE[Y²]dt
    """
    
    def __init__(self,
                 mu: float = 0.05,
                 sigma: float = 0.20,
                 lambda_jump: float = 0.1,
                 mu_jump: float = -0.05,
                 sigma_jump: float = 0.10,
                 climate_beta: float = 0.0,
                 random_state: int = 42):
        """
        Initialize the jump-diffusion model.
        
        Parameters:
        -----------
        mu : float
            Annual drift rate (default: 5%)
        sigma : float
            Annual volatility (default: 20%)
        lambda_jump : float
            Base jump intensity per year (default: 0.1)
        mu_jump : float
            Mean log jump size (default: -5%, negative for downward bias)
        sigma_jump : float
            Jump size volatility (default: 10%)
        climate_beta : float
            Climate sensitivity parameter (default: 0, no climate effect)
        random_state : int
            Random seed for reproducibility
        """
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.climate_beta = climate_beta
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Model state
        self.is_fitted = False
        self.simulation_results = {}
        
        # Set random seed
        np.random.seed(random_state)
        
        # Calculate derived parameters
        self.kappa = np.exp(mu_jump + sigma_jump**2 / 2) - 1  # Mean jump size
        self.drift_adjustment = mu - sigma**2/2 - lambda_jump * self.kappa
        
        self.logger.info(f"JumpDiffusionModel initialized: μ={mu:.3f}, σ={sigma:.3f}, λ={lambda_jump:.3f}")
    
    def simulate_path(self,
                     T: float,
                     n_steps: int,
                     S0: float = 100.0,
                     climate_index: Optional[np.ndarray] = None,
                     return_components: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Simulate a single jump-diffusion path.
        
        Parameters:
        -----------
        T : float
            Time horizon (in years)
        n_steps : int
            Number of time steps
        S0 : float
            Initial asset price
        climate_index : np.ndarray, optional
            Climate index values (length n_steps) to modify jump intensity
        return_components : bool
            If True, return detailed components of the simulation
            
        Returns:
        --------
        prices : np.ndarray
            Simulated price path
        components : dict (if return_components=True)
            Detailed simulation components
            
        Mathematical Implementation:
        ---------------------------
        S_{t+dt} = S_t × exp[(μ - σ²/2 - λκ)dt + σ√dt × Z + J]
        
        where:
        - Z ~ N(0,1) is the Brownian increment
        - J is the compound Poisson jump component
        """
        
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        # Initialize arrays
        prices = np.zeros(n_steps + 1)
        prices[0] = S0
        
        # Storage for components (if requested)
        if return_components:
            components = {
                'times': times,
                'drift_component': np.zeros(n_steps),
                'diffusion_component': np.zeros(n_steps),
                'jump_component': np.zeros(n_steps),
                'jump_times': [],
                'jump_sizes': [],
                'effective_lambda': np.zeros(n_steps)
            }
        
        # Generate random numbers
        brownian_increments = np.random.normal(0, np.sqrt(dt), n_steps)
        
        for i in range(n_steps):
            
            # Calculate climate-adjusted jump intensity
            if climate_index is not None and i < len(climate_index):
                lambda_effective = self.lambda_jump + self.climate_beta * climate_index[i]
                lambda_effective = max(0, lambda_effective)  # Ensure non-negative
            else:
                lambda_effective = self.lambda_jump
            
            if return_components:
                components['effective_lambda'][i] = lambda_effective
            
            # Drift component
            drift = self.drift_adjustment * dt
            
            # Diffusion component  
            diffusion = self.sigma * brownian_increments[i]
            
            # Jump component
            n_jumps = np.random.poisson(lambda_effective * dt)
            jump_total = 0.0
            
            if n_jumps > 0:
                # Generate jump sizes
                jump_sizes = np.random.normal(self.mu_jump, self.sigma_jump, n_jumps)
                jump_total = np.sum(jump_sizes)
                
                if return_components:
                    components['jump_times'].extend([times[i+1]] * n_jumps)
                    components['jump_sizes'].extend(jump_sizes)
            
            # Store components
            if return_components:
                components['drift_component'][i] = drift
                components['diffusion_component'][i] = diffusion  
                components['jump_component'][i] = jump_total
            
            # Update price using log-normal dynamics
            log_return = drift + diffusion + jump_total
            prices[i + 1] = prices[i] * np.exp(log_return)
        
        if return_components:
            return prices, components
        else:
            return prices
    
    def simulate_paths(self,
                      T: float,
                      n_steps: int,
                      n_paths: int,
                      S0: float = 100.0,
                      climate_index: Optional[np.ndarray] = None,
                      parallel: bool = False) -> np.ndarray:
        """
        Simulate multiple jump-diffusion paths.
        
        Parameters:
        -----------
        T : float
            Time horizon (in years)
        n_steps : int
            Number of time steps per path
        n_paths : int
            Number of paths to simulate
        S0 : float
            Initial asset price
        climate_index : np.ndarray, optional
            Climate index values to modify jump intensity
        parallel : bool
            Use parallel processing (placeholder for future implementation)
            
        Returns:
        --------
        np.ndarray
            Simulated paths array (n_paths × n_steps+1)
        """
        
        self.logger.info(f"Simulating {n_paths} paths over {T} years with {n_steps} steps")
        
        paths = np.zeros((n_paths, n_steps + 1))
        
        for i in range(n_paths):
            # Reset random seed for each path to ensure independence
            np.random.seed(self.random_state + i)
            
            paths[i] = self.simulate_path(T, n_steps, S0, climate_index)
            
            if (i + 1) % 1000 == 0:
                self.logger.debug(f"Completed {i + 1}/{n_paths} paths")
        
        # Store simulation metadata
        self.simulation_results = {
            'n_paths': n_paths,
            'n_steps': n_steps,
            'T': T,
            'S0': S0,
            'timestamp': pd.Timestamp.now(),
            'has_climate_effects': climate_index is not None
        }
        
        return paths
    
    def calculate_option_price(self,
                              S0: float,
                              K: float,
                              T: float,
                              r: float,
                              option_type: str = 'call',
                              n_simulations: int = 100000,
                              climate_index: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate European option price using Monte Carlo simulation.
        
        Parameters:
        -----------
        S0 : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (years)
        r : float
            Risk-free interest rate
        option_type : str
            'call' or 'put'
        n_simulations : int
            Number of Monte Carlo simulations
        climate_index : np.ndarray, optional
            Climate index for climate-dependent jumps
            
        Returns:
        --------
        Dict[str, float]
            Option pricing results including price, Greeks, and confidence intervals
            
        Mathematical Framework:
        ----------------------
        Option Price = e^(-rT) × E[max(S_T - K, 0)] for calls
        Option Price = e^(-rT) × E[max(K - S_T, 0)] for puts
        
        where S_T follows the jump-diffusion process
        """
        
        self.logger.info(f"Calculating {option_type} option price with {n_simulations} simulations")
        
        # Determine number of time steps (daily for accurate simulation)
        n_steps = max(int(T * 252), 50)  # Daily steps, minimum 50
        
        # Simulate final prices
        paths = self.simulate_paths(T, n_steps, n_simulations, S0, climate_index)
        final_prices = paths[:, -1]
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - final_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Discount payoffs
        discounted_payoffs = payoffs * np.exp(-r * T)
        
        # Calculate statistics
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        # Confidence intervals
        confidence_95 = stats.norm.interval(0.95, loc=option_price, scale=std_error)
        confidence_99 = stats.norm.interval(0.99, loc=option_price, scale=std_error)
        
        results = {
            'option_price': option_price,
            'standard_error': std_error,
            'confidence_95_lower': confidence_95[0],
            'confidence_95_upper': confidence_95[1],
            'confidence_99_lower': confidence_99[0],
            'confidence_99_upper': confidence_99[1],
            'min_payoff': np.min(payoffs),
            'max_payoff': np.max(payoffs),
            'mean_final_price': np.mean(final_prices),
            'std_final_price': np.std(final_prices)
        }
        
        self.logger.info(f"Option price calculated: {option_price:.4f} ± {std_error:.4f}")
        
        return results
    
    def calculate_greeks(self,
                        S0: float,
                        K: float,
                        T: float,
                        r: float,
                        option_type: str = 'call',
                        n_simulations: int = 50000,
                        bump_size: float = 0.01) -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences.
        
        Parameters:
        -----------
        S0 : float
            Current stock price
        K : float
            Strike price  
        T : float
            Time to maturity
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
        n_simulations : int
            Number of simulations per Greek calculation
        bump_size : float
            Relative bump size for finite differences
            
        Returns:
        --------
        Dict[str, float]
            Greek values (delta, gamma, theta, rho, vega)
        """
        
        self.logger.info("Calculating option Greeks using finite differences")
        
        # Base option price
        base_price = self.calculate_option_price(S0, K, T, r, option_type, n_simulations)['option_price']
        
        # Delta: ∂V/∂S
        dS = S0 * bump_size
        price_up = self.calculate_option_price(S0 + dS, K, T, r, option_type, n_simulations)['option_price']
        price_down = self.calculate_option_price(S0 - dS, K, T, r, option_type, n_simulations)['option_price']
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma: ∂²V/∂S²
        gamma = (price_up - 2 * base_price + price_down) / (dS ** 2)
        
        # Theta: ∂V/∂T (negative of time decay)
        dT = T * bump_size
        if T > dT:
            price_theta = self.calculate_option_price(S0, K, T - dT, r, option_type, n_simulations)['option_price']
            theta = -(price_theta - base_price) / dT  # Negative for time decay
        else:
            theta = 0.0
        
        # Rho: ∂V/∂r
        dr = bump_size
        price_rho = self.calculate_option_price(S0, K, T, r + dr, option_type, n_simulations)['option_price']
        rho = (price_rho - base_price) / dr
        
        # Vega: ∂V/∂σ (sensitivity to volatility)
        # Temporarily increase volatility
        original_sigma = self.sigma
        self.sigma = original_sigma * (1 + bump_size)
        price_vega = self.calculate_option_price(S0, K, T, r, option_type, n_simulations)['option_price']
        self.sigma = original_sigma  # Reset
        
        vega = (price_vega - base_price) / (original_sigma * bump_size)
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'rho': rho,
            'vega': vega
        }
        
        self.logger.info("Greeks calculation completed")
        
        return greeks
    
    def fit_to_data(self,
                   returns: np.ndarray,
                   climate_index: Optional[np.ndarray] = None,
                   method: str = 'mle') -> Dict[str, float]:
        """
        Fit jump-diffusion parameters to observed return data.
        
        Parameters:
        -----------
        returns : np.ndarray
            Observed log returns
        climate_index : np.ndarray, optional
            Climate index values corresponding to returns
        method : str
            Estimation method ('mle' for Maximum Likelihood)
            
        Returns:
        --------
        Dict[str, float]
            Fitted parameters and goodness-of-fit statistics
            
        Mathematical Framework:
        ----------------------
        Likelihood function for jump-diffusion:
        L(θ) = ∏_{t=1}^T f(r_t | θ)
        
        where f(r_t | θ) is the density of the jump-diffusion return distribution
        """
        
        self.logger.info(f"Fitting jump-diffusion model to {len(returns)} return observations")
        
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]  # Remove NaN values
        
        if len(returns) < 50:
            raise ValueError("Insufficient data for parameter estimation (minimum 50 observations)")
        
        # Initial parameter estimates
        mu_init = np.mean(returns) * 252  # Annualized
        sigma_init = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Identify potential jumps (returns beyond 3 standard deviations)
        threshold = 3 * np.std(returns)
        jump_candidates = np.abs(returns) > threshold
        
        if np.any(jump_candidates):
            lambda_init = np.sum(jump_candidates) / len(returns) * 252  # Annualized
            jump_returns = returns[jump_candidates]
            mu_jump_init = np.mean(jump_returns)
            sigma_jump_init = np.std(jump_returns)
        else:
            lambda_init = 0.1
            mu_jump_init = -0.05
            sigma_jump_init = 0.1
        
        # Define objective function (negative log-likelihood)
        def neg_log_likelihood(params):
            mu, sigma, lambda_jump, mu_jump, sigma_jump = params
            
            # Parameter bounds checking
            if sigma <= 0 or lambda_jump < 0 or sigma_jump <= 0:
                return np.inf
            
            # Calculate log-likelihood (simplified approximation)
            # For exact MLE, would need to implement the full jump-diffusion density
            n = len(returns)
            
            # Approximate likelihood assuming normal returns with occasional jumps
            normal_component = -0.5 * n * np.log(2 * np.pi * sigma**2) - \
                              0.5 * np.sum((returns - mu/252)**2) / (sigma**2/252)
            
            # Jump penalty (simplified)
            jump_penalty = -lambda_jump  # Penalize high jump intensity
            
            return -(normal_component + jump_penalty)
        
        # Optimization bounds
        bounds = [
            (-0.5, 0.5),      # mu (annual drift)
            (0.01, 2.0),      # sigma (annual volatility)
            (0.0, 5.0),       # lambda_jump
            (-0.5, 0.5),      # mu_jump
            (0.01, 1.0)       # sigma_jump
        ]
        
        # Initial guess
        x0 = [mu_init, sigma_init, lambda_init, mu_jump_init, sigma_jump_init]
        
        try:
            # Optimize parameters
            result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds)
            
            if result.success:
                mu_fit, sigma_fit, lambda_fit, mu_jump_fit, sigma_jump_fit = result.x
                
                # Update model parameters
                self.mu = mu_fit
                self.sigma = sigma_fit
                self.lambda_jump = lambda_fit
                self.mu_jump = mu_jump_fit
                self.sigma_jump = sigma_jump_fit
                
                # Recalculate derived parameters
                self.kappa = np.exp(mu_jump_fit + sigma_jump_fit**2 / 2) - 1
                self.drift_adjustment = mu_fit - sigma_fit**2/2 - lambda_fit * self.kappa
                
                self.is_fitted = True
                
                # Calculate fit statistics
                log_likelihood = -result.fun
                n_params = 5
                aic = 2 * n_params - 2 * log_likelihood
                bic = np.log(len(returns)) * n_params - 2 * log_likelihood
                
                fit_results = {
                    'mu': mu_fit,
                    'sigma': sigma_fit,
                    'lambda_jump': lambda_fit,
                    'mu_jump': mu_jump_fit,
                    'sigma_jump': sigma_jump_fit,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'convergence': True
                }
                
                self.logger.info(f"Parameter estimation successful. Log-likelihood: {log_likelihood:.4f}")
                
            else:
                self.logger.warning("Parameter optimization failed")
                fit_results = {'convergence': False, 'message': result.message}
        
        except Exception as e:
            self.logger.error(f"Parameter estimation error: {str(e)}")
            fit_results = {'convergence': False, 'error': str(e)}
        
        return fit_results
    
    def plot_simulation_analysis(self,
                               paths: np.ndarray,
                               T: float,
                               figsize: Tuple[int, int] = (15, 12),
                               save_path: str = None) -> plt.Figure:
        """
        Create comprehensive plots for jump-diffusion simulation analysis.
        
        Parameters:
        -----------
        paths : np.ndarray
            Simulated paths (n_paths × n_steps+1)
        T : float
            Time horizon
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Jump-Diffusion Model Analysis', fontsize=16, fontweight='bold')
        
        n_paths, n_steps = paths.shape
        times = np.linspace(0, T, n_steps)
        
        # 1. Sample paths
        ax1 = axes[0, 0]
        n_plot = min(100, n_paths)  # Plot up to 100 paths
        for i in range(n_plot):
            ax1.plot(times, paths[i], alpha=0.3, linewidth=0.5, color='blue')
        
        # Add mean path
        mean_path = np.mean(paths, axis=0)
        ax1.plot(times, mean_path, 'r-', linewidth=2, label='Mean Path')
        
        ax1.set_title(f'Sample Paths (showing {n_plot} of {n_paths})')
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Final price distribution
        ax2 = axes[0, 1]
        final_prices = paths[:, -1]
        ax2.hist(final_prices, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(final_prices), color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.axvline(np.median(final_prices), color='green', linestyle='--', linewidth=2, label='Median')
        ax2.set_title(f'Final Price Distribution (T={T})')
        ax2.set_xlabel('Final Price')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Return distribution (log returns)
        ax3 = axes[0, 2]
        log_returns = np.log(final_prices / paths[:, 0])
        ax3.hist(log_returns, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
        
        # Overlay normal distribution for comparison
        x_norm = np.linspace(log_returns.min(), log_returns.max(), 100)
        normal_fit = stats.norm.pdf(x_norm, np.mean(log_returns), np.std(log_returns))
        ax3.plot(x_norm, normal_fit, 'r-', linewidth=2, label='Normal Fit')
        
        ax3.set_title('Log Return Distribution')
        ax3.set_xlabel('Log Return')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Path evolution statistics
        ax4 = axes[1, 0]
        percentiles = [5, 25, 50, 75, 95]
        path_stats = np.percentile(paths, percentiles, axis=0)
        
        colors = ['red', 'orange', 'green', 'orange', 'red']
        labels = ['5th', '25th', '50th (Median)', '75th', '95th']
        
        for i, (perc, color, label) in enumerate(zip(path_stats, colors, labels)):
            ax4.plot(times, perc, color=color, linewidth=2, label=f'{label} percentile')
        
        ax4.set_title('Path Evolution Percentiles')
        ax4.set_xlabel('Time (years)')
        ax4.set_ylabel('Price')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Volatility evolution (rolling standard deviation)
        ax5 = axes[1, 1]
        window = max(10, n_steps // 20)  # Rolling window
        rolling_vol = pd.DataFrame(paths.T).rolling(window=window, axis=0).std()
        vol_mean = rolling_vol.mean(axis=1)
        vol_std = rolling_vol.std(axis=1)
        
        ax5.plot(times, vol_mean, 'b-', linewidth=2, label='Mean Volatility')
        ax5.fill_between(times, vol_mean - vol_std, vol_mean + vol_std, 
                        alpha=0.3, color='blue', label='±1 Std Dev')
        ax5.set_title(f'Rolling Volatility (window={window})')
        ax5.set_xlabel('Time (years)')
        ax5.set_ylabel('Volatility')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate summary statistics
        initial_price = paths[0, 0]
        final_mean = np.mean(final_prices)
        final_std = np.std(final_prices)
        total_return_mean = (final_mean / initial_price - 1) * 100
        annual_return = (final_mean / initial_price) ** (1/T) - 1
        annual_vol = np.std(log_returns) / np.sqrt(T)
        
        summary_stats = [
            ['Initial Price', f'{initial_price:.2f}'],
            ['Final Mean Price', f'{final_mean:.2f}'],
            ['Final Std Price', f'{final_std:.2f}'],
            ['Total Return (%)', f'{total_return_mean:.2f}%'],
            ['Annualized Return', f'{annual_return:.2%}'],
            ['Annualized Volatility', f'{annual_vol:.2%}'],
            ['Number of Paths', f'{n_paths:,}'],
            ['Time Horizon', f'{T:.2f} years']
        ]
        
        # Create table
        table = ax6.table(cellText=summary_stats,
                         colLabels=['Statistic', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        ax6.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def get_model_summary(self) -> Dict[str, any]:
        """
        Get comprehensive model summary and parameters.
        
        Returns:
        --------
        Dict[str, any]
            Model summary including parameters and derived statistics
        """
        
        summary = {
            'model_type': 'Jump-Diffusion (Merton)',
            'parameters': {
                'mu': self.mu,
                'sigma': self.sigma,
                'lambda_jump': self.lambda_jump,
                'mu_jump': self.mu_jump,
                'sigma_jump': self.sigma_jump,
                'climate_beta': self.climate_beta
            },
            'derived_parameters': {
                'kappa': self.kappa,
                'drift_adjustment': self.drift_adjustment,
                'expected_jumps_per_year': self.lambda_jump,
                'expected_jump_size': np.exp(self.mu_jump) - 1
            },
            'model_properties': {
                'has_climate_effects': self.climate_beta != 0,
                'is_fitted': self.is_fitted,
                'random_state': self.random_state
            }
        }
        
        if self.simulation_results:
            summary['last_simulation'] = self.simulation_results
        
        return summary
