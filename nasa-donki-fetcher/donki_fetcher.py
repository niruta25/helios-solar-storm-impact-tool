#!/usr/bin/env python3
"""
NASA DONKI API Data Fetcher
Fetches Solar Flares (FLR) and Coronal Mass Ejections (CME) data
"""

import requests
import json
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import time

class DONKIFetcher:
    """Fetches data from NASA's DONKI API"""
    
    BASE_URL = "https://api.nasa.gov/DONKI"
    
    def __init__(self, api_key: str = "DEMO_KEY"):
        """
        Initialize DONKI API fetcher
        
        Args:
            api_key: NASA API key (default: DEMO_KEY for limited requests)
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.data_dir = "data"
        self.setup_logging()
        self.create_data_directory()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('donki_fetcher.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/flr", exist_ok=True)
        os.makedirs(f"{self.data_dir}/cme", exist_ok=True)
    
    def make_request(self, endpoint: str, params: Dict[str, Any]) -> List[Dict]:
        """
        Make API request with error handling and rate limiting
        
        Args:
            endpoint: API endpoint (FLR or CME)
            params: Request parameters
            
        Returns:
            List of data records
        """
        url = f"{self.BASE_URL}/{endpoint}"
        params['api_key'] = self.api_key
        
        try:
            self.logger.info(f"Fetching data from {endpoint} endpoint...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"Successfully fetched {len(data)} records from {endpoint}")
            
            # Rate limiting - be nice to NASA's servers
            time.sleep(1)
            
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data from {endpoint}: {e}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON response from {endpoint}: {e}")
            return []
    
    def fetch_solar_flares(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        Fetch Solar Flare data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of solar flare records
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'startDate': start_date,
            'endDate': end_date
        }
        
        return self.make_request('FLR', params)
    
    def fetch_cme_data(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        Fetch Coronal Mass Ejection data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of CME records
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'startDate': start_date,
            'endDate': end_date
        }
        
        return self.make_request('CME', params)
    
    def save_to_json(self, data: List[Dict], filename: str):
        """Save data to JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Data saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving data to {filepath}: {e}")
    
    def save_to_csv(self, data: List[Dict], filename: str):
        """Save data to CSV file with flattened structure"""
        if not data:
            self.logger.warning(f"No data to save for {filename}")
            return
        
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            # Flatten nested data for CSV format
            flattened_data = []
            for record in data:
                flat_record = self.flatten_dict(record)
                flattened_data.append(flat_record)
            
            # Get all unique keys for CSV headers
            all_keys = set()
            for record in flattened_data:
                all_keys.update(record.keys())
            
            fieldnames = sorted(list(all_keys))
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
            
            self.logger.info(f"Data saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving data to {filepath}: {e}")
    
    def flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV output"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to string representation
                items.append((new_key, json.dumps(v) if v else ''))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def fetch_and_save_all_data(self, start_date: str = None, end_date: str = None):
        """
        Fetch and save both Solar Flare and CME data in multiple formats
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Fetch Solar Flares
        self.logger.info("Starting Solar Flare data fetch...")
        flr_data = self.fetch_solar_flares(start_date, end_date)
        
        if flr_data:
            self.save_to_json(flr_data, f"flr/solar_flares_{timestamp}.json")
            self.save_to_csv(flr_data, f"flr/solar_flares_{timestamp}.csv")
        
        # Fetch CME Data
        self.logger.info("Starting CME data fetch...")
        cme_data = self.fetch_cme_data(start_date, end_date)
        
        if cme_data:
            self.save_to_json(cme_data, f"cme/cme_data_{timestamp}.json")
            self.save_to_csv(cme_data, f"cme/cme_data_{timestamp}.csv")
        
        # Create summary
        summary = {
            'fetch_timestamp': timestamp,
            'date_range': {
                'start_date': start_date or (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'end_date': end_date or datetime.now().strftime('%Y-%m-%d')
            },
            'records_count': {
                'solar_flares': len(flr_data),
                'cme_events': len(cme_data)
            }
        }
        
        self.save_to_json(summary, f"fetch_summary_{timestamp}.json")
        self.logger.info(f"Data fetch completed. Summary: {summary}")

def main():
    """Main execution function"""
    # You can replace 'DEMO_KEY' with your actual NASA API key
    # Get your free API key at: https://api.nasa.gov/
    api_key = os.getenv('NASA_API_KEY', 'DEMO_KEY')
    
    fetcher = DONKIFetcher(api_key=api_key)
    
    # Fetch data for the last 30 days (default)
    fetcher.fetch_and_save_all_data()
    
    # Example: Fetch data for specific date range
    # fetcher.fetch_and_save_all_data('2024-01-01', '2024-01-31')

if __name__ == "__main__":
    main()