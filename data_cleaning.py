"""
Data Cleaning & Preprocessing Script
Group 4 - Milestone 2

Handles missing values, outliers, temporal alignment, and feature engineering
for grid emissions and demand data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess grid emissions and demand data"""
    
    def __init__(self):
        self.cleaning_stats = {
            'duplicates_removed': 0,
            'missing_imputed': 0,
            'outliers_corrected': 0,
            'records_flagged': 0
        }
    
    def remove_duplicates(self, df: pd.DataFrame, subset: list) -> pd.DataFrame:
        """Remove duplicate records based on composite key"""
        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep='first')
        removed = initial_count - len(df_clean)
        
        self.cleaning_stats['duplicates_removed'] += removed
        logger.info(f"Removed {removed} duplicate records ({removed/initial_count*100:.2f}%)")
        
        return df_clean
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        time_col: str, 
        value_col: str
    ) -> pd.DataFrame:
        """
        Handle missing values using forward-fill and interpolation
        
        Strategy:
        - Forward fill for gaps < 30 min
        - Linear interpolation for gaps 30 min - 2 hours
        - Flag but preserve gaps > 2 hours for later exclusion
        """
        df = df.sort_values(time_col).reset_index(drop=True)
        
        # Identify missing values
        missing_mask = df[value_col].isna()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            logger.info(f"No missing values in {value_col}")
            return df
        
        logger.info(f"Found {missing_count} missing values in {value_col} ({missing_count/len(df)*100:.2f}%)")
        
        # Calculate time gaps
        df['time_diff'] = df[time_col].diff()
        
        # Forward fill for small gaps (< 30 min)
        df[f'{value_col}_ffill'] = df[value_col].fillna(method='ffill', limit=2)
        
        # Linear interpolation for medium gaps (30 min - 2 hours)
        df[value_col] = df[value_col].interpolate(method='linear', limit=8)
        
        # Create quality flag
        df[f'{value_col}_is_interpolated'] = missing_mask & df[value_col].notna()
        
        # Count how many were successfully imputed
        imputed = df[f'{value_col}_is_interpolated'].sum()
        self.cleaning_stats['missing_imputed'] += imputed
        
        logger.info(f"Imputed {imputed} values in {value_col}")
        
        return df
    
    def detect_and_correct_outliers(
        self,
        df: pd.DataFrame,
        value_col: str,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect and correct outliers
        
        Args:
            df: DataFrame with data
            value_col: Column to check for outliers
            method: 'iqr' or 'zscore'
            threshold: Number of IQRs or standard deviations
        """
        if method == 'iqr':
            Q1 = df[value_col].quantile(0.25)
            Q3 = df[value_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
        elif method == 'zscore':
            mean = df[value_col].mean()
            std = df[value_col].std()
            
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        
        # Identify outliers
        outlier_mask = (df[value_col] < lower_bound) | (df[value_col] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            logger.info(f"Found {outlier_count} outliers in {value_col} ({outlier_count/len(df)*100:.2f}%)")
            
            # Cap outliers at bounds rather than removing
            df[f'{value_col}_original'] = df[value_col].copy()
            df.loc[df[value_col] < lower_bound, value_col] = lower_bound
            df.loc[df[value_col] > upper_bound, value_col] = upper_bound
            
            # Flag corrected values
            df[f'{value_col}_is_outlier'] = outlier_mask
            
            self.cleaning_stats['outliers_corrected'] += outlier_count
        else:
            logger.info(f"No outliers detected in {value_col}")
        
        return df
    
    def correct_impossible_values(
        self,
        df: pd.DataFrame,
        value_col: str,
        min_val: float = None,
        max_val: float = None
    ) -> pd.DataFrame:
        """Correct physically impossible values"""
        
        corrections = 0
        
        if min_val is not None:
            impossible_low = df[value_col] < min_val
            corrections += impossible_low.sum()
            df.loc[impossible_low, value_col] = np.nan
        
        if max_val is not None:
            impossible_high = df[value_col] > max_val
            corrections += impossible_high.sum()
            df.loc[impossible_high, value_col] = np.nan
        
        if corrections > 0:
            logger.info(f"Corrected {corrections} impossible values in {value_col}")
            self.cleaning_stats['outliers_corrected'] += corrections
        
        return df
    
    def print_summary(self):
        """Print cleaning summary statistics"""
        logger.info("\n" + "="*50)
        logger.info("DATA CLEANING SUMMARY")
        logger.info("="*50)
        for key, value in self.cleaning_stats.items():
            logger.info(f"{key}: {value:,}")
        logger.info("="*50 + "\n")


class DataPreprocessor:
    """Advanced preprocessing and feature engineering"""
    
    def __init__(self):
        pass
    
    def standardize_timestamps(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """
        Standardize all timestamps to UTC and create uniform temporal grid
        """
        # Convert to datetime if not already
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        
        # Create uniform 15-minute intervals
        df = df.set_index(time_col)
        df = df.resample('15min').mean()  # Resample to 15-min intervals
        df = df.reset_index()
        
        logger.info("Standardized timestamps to UTC with 15-minute intervals")
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Create time-based features"""
        
        df['hour'] = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['month'] = df[time_col].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season'] = df['month'].apply(self._get_season)
        
        logger.info("Created temporal features (hour, day_of_week, month, season)")
        
        return df
    
    @staticmethod
    def _get_season(month: int) -> str:
        """Map month to season"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        value_col: str,
        windows: list = [24, 168]  # 24 hours, 7 days
    ) -> pd.DataFrame:
        """Create rolling statistics for baseline comparison"""
        
        for window in windows:
            # Rolling mean
            df[f'{value_col}_rolling_mean_{window}h'] = (
                df[value_col].rolling(window=window*4, center=True).mean()  # *4 for 15-min intervals
            )
            
            # Rolling std
            df[f'{value_col}_rolling_std_{window}h'] = (
                df[value_col].rolling(window=window*4, center=True).std()
            )
        
        logger.info(f"Created rolling features for {value_col}")
        
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        value_col: str,
        lags: list = [4, 24, 96]  # 1hr, 6hr, 24hr in 15-min intervals
    ) -> pd.DataFrame:
        """Create lagged features"""
        
        for lag in lags:
            df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
        
        logger.info(f"Created lag features for {value_col}")
        
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: list,
        method: str = 'zscore'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features
        
        Args:
            df: DataFrame
            columns: Columns to normalize
            method: 'zscore' or 'minmax'
        
        Returns:
            Normalized DataFrame and scaling parameters
        """
        scaling_params = {}
        
        for col in columns:
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df[f'{col}_normalized'] = (df[col] - mean) / std
                scaling_params[col] = {'mean': mean, 'std': std}
                
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
                scaling_params[col] = {'min': min_val, 'max': max_val}
        
        logger.info(f"Normalized {len(columns)} columns using {method}")
        
        return df, scaling_params


def main():
    """Main preprocessing pipeline"""
    
    # Load raw data
    logger.info("Loading raw data...")
    emissions_raw = pd.read_csv('data/raw/emissions_raw.csv')
    demand_raw = pd.read_csv('data/raw/demand_raw.csv')
    
    # Initialize cleaners
    cleaner = DataCleaner()
    preprocessor = DataPreprocessor()
    
    # Clean emissions data
    logger.info("\nCleaning emissions data...")
    emissions = cleaner.remove_duplicates(emissions_raw, subset=['timestamp', 'region'])
    emissions = cleaner.handle_missing_values(emissions, 'timestamp', 'value')
    emissions = cleaner.correct_impossible_values(
        emissions, 
        'value', 
        min_val=0, 
        max_val=1500  # Max realistic MOER
    )
    emissions = cleaner.detect_and_correct_outliers(emissions, 'value', method='iqr')
    
    # Clean demand data
    logger.info("\nCleaning demand data...")
    demand = cleaner.remove_duplicates(demand_raw, subset=['timestamp', 'balancing_authority'])
    demand = cleaner.handle_missing_values(demand, 'timestamp', 'demand_MW')
    demand = cleaner.correct_impossible_values(demand, 'demand_MW', min_val=0)
    demand = cleaner.detect_and_correct_outliers(demand, 'demand_MW', method='iqr', threshold=3.5)
    
    # Print cleaning summary
    cleaner.print_summary()
    
    # Preprocess emissions
    logger.info("\nPreprocessing emissions data...")
    emissions = preprocessor.standardize_timestamps(emissions, 'timestamp')
    emissions = preprocessor.create_temporal_features(emissions, 'timestamp')
    emissions = preprocessor.create_rolling_features(emissions, 'value', windows=[24, 168])
    emissions = preprocessor.create_lag_features(emissions, 'value', lags=[4, 24, 96])
    
    # Preprocess demand
    logger.info("\nPreprocessing demand data...")
    demand = preprocessor.standardize_timestamps(demand, 'timestamp')
    demand = preprocessor.create_temporal_features(demand, 'timestamp')
    demand = preprocessor.create_rolling_features(demand, 'demand_MW', windows=[24, 168])
    demand = preprocessor.create_lag_features(demand, 'demand_MW', lags=[4, 24, 96])
    
    # Save cleaned data
    logger.info("\nSaving cleaned data...")
    emissions.to_csv('data/processed/emissions_clean.csv', index=False)
    demand.to_csv('data/processed/demand_clean.csv', index=False)
    
    # Generate data quality report
    logger.info("\nGenerating data quality report...")
    
    report = {
        'emissions': {
            'total_records': len(emissions),
            'date_range': f"{emissions['timestamp'].min()} to {emissions['timestamp'].max()}",
            'mean_value': emissions['value'].mean(),
            'std_value': emissions['value'].std(),
            'missing_pct': emissions['value'].isna().sum() / len(emissions) * 100
        },
        'demand': {
            'total_records': len(demand),
            'date_range': f"{demand['timestamp'].min()} to {demand['timestamp'].max()}",
            'mean_demand': demand['demand_MW'].mean(),
            'std_demand': demand['demand_MW'].std(),
            'missing_pct': demand['demand_MW'].isna().sum() / len(demand) * 100
        }
    }
    
    import json
    with open('data/processed/data_quality_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("Preprocessing complete!")
    logger.info(f"Cleaned emissions records: {len(emissions):,}")
    logger.info(f"Cleaned demand records: {len(demand):,}")


if __name__ == "__main__":
    main()
