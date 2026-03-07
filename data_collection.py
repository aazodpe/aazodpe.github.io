"""
Data Collection Script for Grid Emissions & Demand Analysis
Group 4 - Milestone 2

This script collects data from WattTime API and EIA-930 API for analyzing
hazardous grid events and their impact on electricity demand and emissions.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WattTimeCollector:
    """
    Collector for WattTime API - Marginal Operating Emissions Rate (MOER)
    """
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.base_url = "https://api.watttime.org/v3"
        self.token = None
        self.token_expiry = None
        
    def authenticate(self):
        """Authenticate and get access token"""
        # v3/login returns HTML; use v2/login which returns the JWT token
        login_url = "https://api.watttime.org/v2/login"
        response = requests.get(login_url, auth=(self.username, self.password))
        
        if response.status_code == 200:
            self.token = response.json()['token']
            self.token_expiry = datetime.now() + timedelta(minutes=30)
            logger.info("WattTime authentication successful")
        else:
            raise Exception(f"Authentication failed: {response.text}")
    
    def get_historical_data(
        self, 
        region: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical MOER data for a specific region and time range
        
        Args:
            region: Grid region identifier (e.g., 'CAISO', 'PSCO')
            start_time: Start datetime (UTC)
            end_time: End datetime (UTC)
            
        Returns:
            DataFrame with timestamp, region, and emissions data
        """
        # Check if token needs refresh
        if not self.token or datetime.now() >= self.token_expiry:
            self.authenticate()
        
        headers = {'Authorization': f'Bearer {self.token}'}
        def to_utc_str(dt):
            if dt.tzinfo is None:
                return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            return dt.isoformat()

        params = {
            'region': region,
            'start': to_utc_str(start_time),
            'end': to_utc_str(end_time),
            'signal_type': 'co2_moer'
        }
        
        historical_url = f"{self.base_url}/historical"
        response = requests.get(historical_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'])
            df['region'] = region
            logger.info(f"Collected {len(df)} records for {region}")
            return df
        else:
            logger.error(f"Data collection failed: {response.text}")
            return pd.DataFrame()
    
    def collect_multiple_events(
        self, 
        region: str, 
        event_windows: List[Dict]
    ) -> pd.DataFrame:
        """
        Collect data for multiple event windows with rate limiting
        
        Args:
            region: Grid region identifier
            event_windows: List of dicts with 'start' and 'end' datetime objects
            
        Returns:
            Combined DataFrame for all events
        """
        all_data = []
        
        for i, window in enumerate(event_windows):
            logger.info(f"Collecting event {i+1}/{len(event_windows)}")
            
            # Add buffer before and after event for baseline comparison
            buffer = timedelta(days=7)
            start = window['start'] - buffer
            end = window['end'] + buffer
            
            df = self.get_historical_data(region, start, end)
            df['event_id'] = window.get('event_id', f'event_{i}')
            all_data.append(df)
            
            # Rate limiting - respect API limits
            time.sleep(1)
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total records collected: {len(combined)}")
        return combined


class EIACollector:
    """
    Collector for EIA-930 API - Hourly Electricity Demand Data
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    
    def get_demand_data(
        self,
        balancing_authority: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch hourly demand data from EIA-930
        
        Args:
            balancing_authority: BA identifier (e.g., 'CISO', 'PSCO')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with timestamp and demand data
        """
        all_data = []
        offset = 0

        while True:
            # EIA v2 requires data[] list syntax passed as tuples to preserve key
            params = [
                ('api_key', self.api_key),
                ('frequency', 'hourly'),
                ('data[]', 'value'),
                ('facets[respondent][]', balancing_authority),
                ('facets[type][]', 'D'),
                ('start', start_date),
                ('end', end_date),
                ('sort[0][column]', 'period'),
                ('sort[0][direction]', 'asc'),
                ('offset', offset),
                ('length', 5000),
            ]
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                result = response.json()
                data = result['response']['data']
                
                if not data:
                    break
                
                all_data.extend(data)
                offset += len(data)
                
                logger.info(f"Fetched {len(all_data)} records so far...")
                
                # Check if we've reached the end
                if len(data) < 5000:
                    break
                    
                time.sleep(0.5)  # Rate limiting
            else:
                logger.error(f"EIA API error: {response.text}")
                break
        
        df = pd.DataFrame(all_data)
        logger.info(f"Total EIA records collected: {len(df)}")
        return df
    
    def process_demand_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw EIA data into clean format
        
        Args:
            df: Raw DataFrame from EIA API
            
        Returns:
            Processed DataFrame with standardized columns
        """
        # Rename columns to match expected schema
        processed = df.rename(columns={
            'period': 'timestamp',
            'value': 'demand_MW',
            'respondent': 'balancing_authority'
        })

        # Convert timestamp to datetime
        processed['timestamp'] = pd.to_datetime(processed['timestamp'])
        processed['demand_MW'] = pd.to_numeric(processed['demand_MW'], errors='coerce')

        # Keep only relevant columns
        processed = processed[['timestamp', 'balancing_authority', 'demand_MW']]
        
        return processed


def create_event_catalog() -> pd.DataFrame:
    """
    Create catalog of hazardous grid events
    
    This function would typically read from curated sources (utility reports,
    CPUC filings, etc.). Here we show the data structure.
    
    Returns:
        DataFrame with event metadata
    """
    events = [
        {
            'event_id': 'psps_boulder_202412',
            'event_type': 'PSPS',
            'start': datetime(2024, 12, 10, 19, 0),  # UTC
            'end': datetime(2024, 12, 11, 6, 0),
            'region': 'PSCO',
            'severity': 'moderate',
            'affected_customers': 8500,
            'description': 'Boulder area PSPS due to high wind forecast'
        },
        {
            'event_id': 'weather_ca_202408',
            'event_type': 'extreme_weather',
            'start': datetime(2024, 8, 15, 0, 0),
            'end': datetime(2024, 8, 18, 23, 59),
            'region': 'CAISO',
            'severity': 'high',
            'affected_customers': 125000,
            'description': 'Heat wave causing grid stress and rolling outages'
        },
        # Add more events...
    ]
    
    return pd.DataFrame(events)


def main():
    """Main execution function"""
    
    # Configuration — loaded from .env
    WATTTIME_USER = os.getenv("WATTTIME_USERNAME")
    WATTTIME_PASS = os.getenv("WATTTIME_PASSWORD")
    EIA_API_KEY = os.getenv("EIA_API_KEY")
    
    # Initialize collectors
    watttime = WattTimeCollector(WATTTIME_USER, WATTTIME_PASS)
    eia = EIACollector(EIA_API_KEY)
    
    # Load event catalog
    events = create_event_catalog()
    logger.info(f"Loaded {len(events)} events")
    
    # Only collect for CAISO_NORTH (account access is limited to this region)
    caiso_events = events[events['region'] == 'CAISO']
    logger.info(f"Collecting WattTime data for {len(caiso_events)} CAISO event(s)")

    emissions_data = watttime.collect_multiple_events(
        region='CAISO_NORTH',
        event_windows=[
            {'start': row['start'], 'end': row['end'], 'event_id': row['event_id']}
            for _, row in caiso_events.iterrows()
        ]
    )

    # Collect demand data (requires EIA API key)
    demand_data = pd.DataFrame()
    if EIA_API_KEY and EIA_API_KEY != 'your_eia_api_key':
        demand_data_list = []
        for _, event in caiso_events.iterrows():
            start_date = (event['start'] - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = (event['end'] + timedelta(days=7)).strftime('%Y-%m-%d')

            demand = eia.get_demand_data(
                balancing_authority='CISO',
                start_date=start_date,
                end_date=end_date
            )
            demand['event_id'] = event['event_id']
            demand_data_list.append(demand)

        if demand_data_list:
            demand_data = pd.concat(demand_data_list, ignore_index=True)
            demand_data = eia.process_demand_data(demand_data)
    else:
        logger.warning("EIA_API_KEY not set — skipping demand data collection")

    # Save raw data
    emissions_data.to_csv('data/raw/emissions_raw.csv', index=False)
    if not demand_data.empty:
        demand_data.to_csv('data/raw/demand_raw.csv', index=False)
    events.to_csv('data/raw/events_catalog.csv', index=False)
    
    logger.info("Data collection complete!")
    logger.info(f"Emissions records: {len(emissions_data)}")
    logger.info(f"Demand records: {len(demand_data)}")
    logger.info(f"Events cataloged: {len(events)}")


if __name__ == "__main__":
    main()
