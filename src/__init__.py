"""
Multi-Regime Climate-Financial Risk Transmission Engine
A PhD-level framework for modeling climate-financial risk transmission.
"""

__version__ = "1.0.0"
__author__ = "Climate Risk Research Team"
__email__ = "research@climaterisk.org"

import logging
import os

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'climate_risk_engine.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Multi-Regime Climate-Financial Risk Transmission Engine initialized")
