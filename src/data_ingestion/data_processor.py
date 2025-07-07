"""
Data Processing Module
Handles data preprocessing, cleaning, and storage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path
import h5py
import json
from datetime import datetime, timedelta

class DataProcessor:
    """Handles data preprocessing and cleaning."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_financial_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean financial data by handling missing values and outliers.
        
        Args:
            data: Raw financial data
            
        Returns:
            Cleaned financial data
        """
        cleaned_data = data.copy()
        
        # Handle missing values
        cleaned_data = cleaned_data.fillna(method='ffill')
        cleaned_data = cleaned_data.fillna(method='bfill')
        
        # Remove extreme outliers (more than 5 standard deviations)
        for col in cleaned_data.select_dtypes(include=[np.number]).columns:
            mean = cleaned_data[col].mean()
            std = cleaned_data[col].std()
            outlier_mask = np.abs(cleaned_data[col] - mean) > 5 * std
            cleaned_data.loc[outlier_mask, col] = np.nan
        
        # Fill remaining NaN values
        cleaned_data = cleaned_data.fillna(method='ffill')
        
        return cleaned_data
    
    def calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            price_data: Price data
            
        Returns:
            DataFrame with returns
        """
        returns = price_data.pct_change().dropna()
        return returns
    
    def calculate_volatility(self, returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Calculate rolling volatility.
        
        Args:
            returns: Returns data
            window: Rolling window size
            
        Returns:
            DataFrame with volatility
        """
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility
    
    def align_time_series(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align multiple time series to common dates.
        
        Args:
            data_dict: Dictionary of time series data
            
        Returns:
            Dictionary with aligned time series
        """
        # Find common date range
        min_date = None
        max_date = None
        
        for key, df in data_dict.items():
            if 'date' in df.columns:
                date_col = 'date'
            else:
                date_col = df.index if isinstance(df.index, pd.DatetimeIndex) else None
            
            if date_col is not None:
                if isinstance(date_col, str):
                    dates = df[date_col]
                else:
                    dates = date_col
                
                if min_date is None or dates.min() > min_date:
                    min_date = dates.min()
                if max_date is None or dates.max() < max_date:
                    max_date = dates.max()
        
        # Align all series to common date range
        aligned_data = {}
        for key, df in data_dict.items():
            if 'date' in df.columns:
                aligned_df = df[
                    (df['date'] >= min_date) & (df['date'] <= max_date)
                ].copy()
            else:
                aligned_df = df.loc[min_date:max_date].copy()
            
            aligned_data[key] = aligned_df
        
        return aligned_data
    
    def create_feature_matrix(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create feature matrix from multiple data sources.
        
        Args:
            data_dict: Dictionary of data sources
            
        Returns:
            Combined feature matrix
        """
        features = []
        
        for key, df in data_dict.items():
            if df.empty:
                continue
                
            # Prepare dataframe
            df_processed = df.copy()
            
            # Ensure date column
            if 'date' not in df_processed.columns:
                if isinstance(df_processed.index, pd.DatetimeIndex):
                    df_processed['date'] = df_processed.index
                else:
                    continue
            
            # Select numeric columns
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            
            # Rename columns with prefix
            df_processed = df_processed[['date'] + list(numeric_cols)]
            df_processed.columns = ['date'] + [f"{key}_{col}" for col in numeric_cols]
            
            features.append(df_processed)
        
        if not features:
            return pd.DataFrame()
        
        # Merge all features
        feature_matrix = features[0]
        for df in features[1:]:
            feature_matrix = pd.merge(feature_matrix, df, on='date', how='outer')
        
        # Sort by date
        feature_matrix = feature_matrix.sort_values('date')
        
        return feature_matrix
    
    def save_to_hdf5(self, data: pd.DataFrame, filename: str, group: str = 'data'):
        """
        Save data to HDF5 format.
        
        Args:
            data: Data to save
            filename: Output filename
            group: HDF5 group name
        """
        filepath = self.processed_dir / f"{filename}.h5"
        
        try:
            with h5py.File(filepath, 'w') as f:
                # Save data
                grp = f.create_group(group)
                
                # Save each column
                for col in data.columns:
                    if data[col].dtype == 'object':
                        # Convert object columns to string
                        grp.create_dataset(col, data=data[col].astype(str))
                    else:
                        grp.create_dataset(col, data=data[col].values)
                
                # Save metadata
                grp.attrs['columns'] = list(data.columns)
                grp.attrs['shape'] = data.shape
                grp.attrs['created'] = datetime.now().isoformat()
                
        except Exception as e:
            self.logger.error(f"Error saving to HDF5: {e}")
    
    def load_from_hdf5(self, filename: str, group: str = 'data') -> pd.DataFrame:
        """
        Load data from HDF5 format.
        
        Args:
            filename: Input filename
            group: HDF5 group name
            
        Returns:
            Loaded DataFrame
        """
        filepath = self.processed_dir / f"{filename}.h5"
        
        try:
            with h5py.File(filepath, 'r') as f:
                grp = f[group]
                
                # Load data
                data = {}
                for col in grp.attrs['columns']:
                    data[col] = grp[col][:]
                
                return pd.DataFrame(data)
                
        except Exception as e:
            self.logger.error(f"Error loading from HDF5: {e}")
            return pd.DataFrame()
    
    def save_to_parquet(self, data: pd.DataFrame, filename: str):
        """
        Save data to Parquet format.
        
        Args:
            data: Data to save
            filename: Output filename
        """
        filepath = self.processed_dir / f"{filename}.parquet"
        
        try:
            data.to_parquet(filepath, compression='snappy')
        except Exception as e:
            self.logger.error(f"Error saving to Parquet: {e}")
    
    def load_from_parquet(self, filename: str) -> pd.DataFrame:
        """
        Load data from Parquet format.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded DataFrame
        """
        filepath = self.processed_dir / f"{filename}.parquet"
        
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            self.logger.error(f"Error loading from Parquet: {e}")
            return pd.DataFrame()
    
    def generate_summary_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for dataset.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_summary': data.describe().to_dict(),
            'date_range': {
                'start': data.index.min() if isinstance(data.index, pd.DatetimeIndex) else None,
                'end': data.index.max() if isinstance(data.index, pd.DatetimeIndex) else None
            }
        }
        
        return stats
