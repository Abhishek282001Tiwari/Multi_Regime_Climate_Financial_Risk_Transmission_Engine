"""
Data Ingestion Module
Handles collection and preprocessing of financial and climate data from FREE sources only.
"""

from .financial_data_collector import FinancialDataCollector

__all__ = ['FinancialDataCollector']
