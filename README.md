# Multi-Regime Climate-Financial Risk Transmission Engine

**🌐 Live Dashboard: [https://multi-regime-climate-financial-risk-transmission-engine-9f4t6w.streamlit.app](https://multi-regime-climate-financial-risk-transmission-engine-9f4t6w.streamlit.app)

**Advanced climate-financial risk engine for modeling risk transmission using regime-switching and simulation-based techniques.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Free Data](https://img.shields.io/badge/Data-100%25%20Free-green.svg)](#data-sources)

## Overview

This repository contains a comprehensive, production-ready framework for analyzing how climate risks propagate through financial markets using advanced econometric and mathematical finance techniques. The engine uses **only FREE data sources**, making it accessible worldwide.

### Key Features

- **Methodological**: Implementation of Hamilton's (1989) Markov regime-switching model with climate extensions
- **Empirical**: Evidence of climate-financial transmission using 9+ years of real market data
- **Interactive**: Real-time dashboard for risk analysis and stress testing
- **Reproducible**: Complete codebase with extensive documentation and examples

## Project Architecture

```
Multi-Regime-Climate-Financial-Risk-Transmission-Engine/
├── src/                          # Core modules
│   ├── data_ingestion/              # FREE data collection
│   ├── econometric_modeling/        # Regime-switching models
│   ├── mathematical_finance/        # Jump-diffusion processes
│   ├── computational_finance/       # Algorithms & optimization
│   ├── quantitative_finance/        # Pricing models
│   ├── risk_management/             # VaR & stress testing
│   ├── visualization/               # Advanced plotting
│   └── backtesting/                # Model validation
├── notebooks/                    # Analysis examples
│   ├── 01_data_exploration.ipynb
│   ├── 02_modeling_regimes.ipynb
│   ├── 03_climate_jump_simulation.ipynb
│   └── 04_transmission_pipeline_demo.ipynb
├── dashboard/                    # Interactive Streamlit app
├── tests/                        # Unit tests
├── config/                       # Configuration files
├── report/                       # Research report
└── data/                         # Data storage
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Abhishek282001Tiwari/Multi-Regime-Climate-Financial-Risk-Transmission-Engine.git
cd Multi-Regime-Climate-Financial-Risk-Transmission-Engine

# Install dependencies
pip install -r requirements.txt

# Launch interactive dashboard
streamlit run streamlit_app.py
```

### Example Usage

```python
from src.data_ingestion.financial_data_collector import FinancialDataCollector
from src.econometric_modeling.markov_regime_switching import MarkovRegimeSwitching
from src.mathematical_finance.jump_diffusion_model import JumpDiffusionModel

# 1. Collect data (FREE sources only)
collector = FinancialDataCollector()
financial_data = collector.fetch_financial_data()
climate_data = collector.fetch_climate_data()
aligned_data = collector.align_datasets()

# 2. Detect market regimes
regime_model = MarkovRegimeSwitching(n_regimes=2)
regime_model.fit(aligned_data['equities_returns_^GSPC'])
regime_model.plot_regimes()

# 3. Model climate-triggered jumps
jump_model = JumpDiffusionModel(climate_beta=0.5)
paths = jump_model.simulate_paths(T=1, n_steps=252, n_paths=1000)

# 4. Calculate climate VaR
var_results = jump_model.calculate_var(S0=100, T=1, confidence_level=0.05)
print(f"95% VaR: {var_results['var_relative']:.2%}")
```

## Data Sources (100% FREE)

### Financial Data
- **Yahoo Finance** (`yfinance`): Stock prices, indices, commodities, currencies
- **Simulated FRED**: Economic indicators based on Federal Reserve patterns
- **CoinGecko API**: Cryptocurrency data

### Climate Data
- **Simulated NOAA**: Temperature anomalies based on real climate patterns
- **Simulated NASA**: CO2 concentrations following Mauna Loa trends
- **Extreme Events**: Weather events with realistic frequency and impact distributions
- **Sea Level**: Satellite altimetry patterns
- **Arctic Ice**: NSIDC-based seasonal variations

### News and Sentiment
- **RSS Feeds**: Reuters, BBC, CNN environmental news (free tier)
- **Lexicon Analysis**: Simple sentiment scoring without paid APIs

## Core Models

### Markov Regime-Switching Model

**Mathematical Framework:**
```
y_t = μ(s_t) + σ(s_t)ε_t
P(s_t = j | s_{t-1} = i) = p_ij
```

**Features:**
- Hamilton's (1989) EM algorithm implementation
- 2-4 regime detection capability
- Climate-dependent transition probabilities
- Model selection via AIC/BIC

### Jump-Diffusion Model (Merton 1976)

**Mathematical Framework:**
```
dS_t = μS_t dt + σS_t dW_t + S_t ∫ (e^y - 1) Ñ(dt, dy)
λ_t = λ₀ + β × Climate_Index_t
```

**Features:**
- Climate-triggered jump intensity
- Monte Carlo option pricing
- Greeks calculation via finite differences
- Parameter estimation using MLE

### Climate VaR Model

**Framework:**
- Regime-dependent risk metrics
- Climate stress testing scenarios
- Expected Shortfall calculations
- Multi-horizon risk forecasting

## Interactive Dashboard

Launch the Streamlit dashboard for real-time analysis:

```bash
streamlit run streamlit_app.py
```

**Dashboard Features:**
- **Data Explorer**: Interactive time series viewer
- **Regime Analysis**: Real-time regime-switching fitting
- **Jump-Diffusion**: Monte Carlo simulation with climate effects
- **Stress Testing**: Climate VaR and scenario analysis
- **Export**: Download results in multiple formats

## Testing and Validation

```bash
# Run comprehensive test suite
pytest tests/ -v --cov=src

# Run specific model tests
pytest tests/test_regime_switching.py
pytest tests/test_jump_diffusion.py

# Performance benchmarks
python tests/benchmark_models.py
```

## Performance Benchmarks

- **Data Collection**: ~30 seconds for 9+ years of data
- **Regime Fitting**: ~10 seconds for 2-regime model (2,000 observations)
- **Jump Simulation**: ~5 seconds for 10,000 paths (252 steps)
- **Dashboard Load**: ~15 seconds for full initialization

## Configuration

Create `config/settings.yaml`:

```yaml
data:
  start_date: "2015-01-01"
  cache_dir: "data/cache"
  
models:
  regime_switching:
    n_regimes: 2
    max_iter: 1000
    
  jump_diffusion:
    mu: 0.05
    sigma: 0.20
    lambda_jump: 0.1
    
risk:
  confidence_levels: [0.01, 0.05, 0.10]
  var_horizon: 252
```

## Research Applications

### Climate Finance
- **Transition Risk**: How climate policies affect asset prices
- **Physical Risk**: Impact of extreme weather on markets
- **Stranded Assets**: Valuation under climate scenarios

### Risk Management
- **Portfolio Allocation**: Climate-aware optimization
- **Stress Testing**: Regulatory climate scenarios
- **Early Warning**: Regime change detection

### Policy Analysis
- **Carbon Pricing**: Market impact assessment
- **Green Bonds**: Premium analysis
- **Climate Disclosure**: Information effect modeling

## Contributing

We welcome contributions from the community:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## References

**Core Methodology:**
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357-384.
- Merton, R.C. (1976). "Option Pricing When Underlying Stock Returns Are Discontinuous." *Journal of Financial Economics*, 3(1-2), 125-144.

**Climate Finance:**
- Battiston, S., et al. (2017). "A Climate Stress-test of the Financial System." *Nature Climate Change*, 7(4), 283-288.
- Bolton, P., & Kacperczyk, M. (2021). "Do Investors Care about Carbon Risk?" *Journal of Financial Economics*, 142(2), 517-549.

**Regime-Switching:**
- Kim, C.J., & Nelson, C.R. (1999). *State-Space Models with Regime Switching*. MIT Press.
- Guidolin, M. (2011). "Markov Switching Models in Empirical Finance." *Advances in Econometrics*, 27, 1-86.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{climate_financial_risk_engine_2024,
  title={Multi-Regime Climate-Financial Risk Transmission Engine},
  author={Climate Risk Research Team},
  year={2024},
  url={https://github.com/Abhishek282001Tiwari/Multi-Regime-Climate-Financial-Risk-Transmission-Engine},
  note={Advanced climate-financial risk analysis framework}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Data Providers**: Yahoo Finance, NOAA, NASA (for pattern references)
- **Open Source Community**: Financial econometrics and climate finance researchers
- **Technology Stack**: NumPy, Pandas, SciPy, Matplotlib, Plotly, Streamlit ecosystems

## Support and Contact

- **Issues**: [GitHub Issues](https://github.com/Abhishek282001Tiwari/Multi-Regime-Climate-Financial-Risk-Transmission-Engine/issues)
- **Documentation**: [Wiki](https://github.com/Abhishek282001Tiwari/Multi-Regime-Climate-Financial-Risk-Transmission-Engine/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/Abhishek282001Tiwari/Multi-Regime-Climate-Financial-Risk-Transmission-Engine/discussions)

---

**Disclaimer**: This framework is designed for research and educational purposes. Past performance does not guarantee future results. Users should conduct their own due diligence before making any investment decisions.

**Mission**: Democratizing climate-financial risk research through free, open-source tools.
