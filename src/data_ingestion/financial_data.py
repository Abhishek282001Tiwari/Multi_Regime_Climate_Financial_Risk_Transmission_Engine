"""
Financial Data Collection Module
Collects financial market data from free sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import time

class FinancialDataCollector:
    """Collects financial market data from free APIs."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
    def fetch_stock_data(self, symbols: List[str], period: str = "2y") -> pd.DataFrame:
        """
        Fetch stock data using yfinance.
        
        Args:
            symbols: List of stock symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with stock data
        """
        try:
            data = yf.download(symbols, period=period, group_by='ticker')
            return data
        except Exception as e:
            self.logger.error(f"Error fetching stock data: {e}")
            return pd.DataFrame()
    
    def fetch_crypto_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Fetch cryptocurrency data from CoinGecko API (free).
        
        Args:
            symbols: List of crypto symbols
            
        Returns:
            DataFrame with crypto data
        """
        if symbols is None:
            symbols = ['bitcoin', 'ethereum', 'cardano', 'solana']
            
        base_url = "https://api.coingecko.com/api/v3"
        crypto_data = []
        
        for symbol in symbols:
            try:
                url = f"{base_url}/coins/{symbol}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': '365',
                    'interval': 'daily'
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                prices = data['prices']
                volumes = data['total_volumes']
                
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['volume'] = [v[1] for v in volumes]
                df['symbol'] = symbol
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                crypto_data.append(df)
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error fetching crypto data for {symbol}: {e}")
                continue
        
        return pd.concat(crypto_data, ignore_index=True) if crypto_data else pd.DataFrame()
    
    def fetch_commodity_data(self) -> pd.DataFrame:
        """
        Fetch commodity data from free sources.
        
        Returns:
            DataFrame with commodity data
        """
        commodities = {
            'GC=F': 'Gold',
            'SI=F': 'Silver',
            'CL=F': 'Crude Oil',
            'NG=F': 'Natural Gas',
            'ZC=F': 'Corn',
            'ZS=F': 'Soybeans'
        }
        
        try:
            data = yf.download(list(commodities.keys()), period="2y")
            return data
        except Exception as e:
            self.logger.error(f"Error fetching commodity data: {e}")
            return pd.DataFrame()
    
    def fetch_currency_data(self) -> pd.DataFrame:
        """
        Fetch currency data from free sources.
        
        Returns:
            DataFrame with currency data
        """
        currencies = ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'AUDUSD=X', 'CADUSD=X']
        
        try:
            data = yf.download(currencies, period="2y")
            return data
        except Exception as e:
            self.logger.error(f"Error fetching currency data: {e}")
            return pd.DataFrame()
    
    def fetch_bond_data(self) -> pd.DataFrame:
        """
        Fetch bond yield data from free sources.
        
        Returns:
            DataFrame with bond data
        """
        bonds = {
            '^TNX': 'US_10Y',
            '^TYX': 'US_30Y',
            '^FVX': 'US_5Y',
            '^IRX': 'US_3M'
        }
        
        try:
            data = yf.download(list(bonds.keys()), period="2y")
            return data
        except Exception as e:
            self.logger.error(f"Error fetching bond data: {e}")
            return pd.DataFrame()
    
    def fetch_esg_etf_data(self) -> pd.DataFrame:
        """
        Fetch ESG/climate-related ETF data.
        
        Returns:
            DataFrame with ESG ETF data
        """
        esg_etfs = {
            'ICLN': 'Clean Energy',
            'QCLN': 'Clean Energy',
            'ESG': 'ESG',
            'KRBN': 'Carbon',
            'GEX': 'Global Environment'
        }
        
        try:
            data = yf.download(list(esg_etfs.keys()), period="2y")
            return data
        except Exception as e:
            self.logger.error(f"Error fetching ESG ETF data: {e}")
            return pd.DataFrame()
    
    def get_comprehensive_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch comprehensive financial market data.
        
        Returns:
            Dictionary containing all financial data
        """
        data = {}
        
        # Major indices
        indices = ['^GSPC', '^DJI', '^IXIC', '^FTSE', '^GDAXI', '^N225']
        data['indices'] = self.fetch_stock_data(indices)
        
        # Commodities
        data['commodities'] = self.fetch_commodity_data()
        
        # Currencies
        data['currencies'] = self.fetch_currency_data()
        
        # Bonds
        data['bonds'] = self.fetch_bond_data()
        
        # ESG ETFs
        data['esg_etfs'] = self.fetch_esg_etf_data()
        
        # Crypto
        data['crypto'] = self.fetch_crypto_data()
        
        return data
