"""
Configuration settings for NASA DONKI Data Fetcher
"""

import os
from datetime import datetime, timedelta

class Config:
    """Configuration class for DONKI fetcher"""
    
    # API Configuration
    NASA_API_KEY = os.getenv('NASA_API_KEY', 'DEMO_KEY')
    DONKI_BASE_URL = "https://api.nasa.gov/DONKI"
    
    # Rate limiting (requests per hour)
    RATE_LIMITS = {
        'DEMO_KEY': 30,
        'PERSONAL': 1000
    }
    
    # Default date ranges (days)
    DEFAULT_LOOKBACK_DAYS = 30
    
    # Data storage
    DATA_DIR = "data"
    LOG_FILE = "donki_fetcher.log"
    
    # File formats to save
    SAVE_FORMATS = ['json', 'csv']
    
    # Request timeout (seconds)
    REQUEST_TIMEOUT = 30
    
    # Delay between API calls (seconds)
    API_DELAY = 1
    
    # Endpoints
    ENDPOINTS = {
        'solar_flares': 'FLR',
        'cme': 'CME',
        'geomagnetic_storms': 'GST',
        'solar_energetic_particles': 'SEP',
        'magnetopause_crossings': 'MPC',
        'radiation_belt_enhancements': 'RBE',
        'hss': 'HSS'  # High Speed Streams
    }
    
    @classmethod
    def get_default_date_range(cls):
        """Get default start and end dates"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=cls.DEFAULT_LOOKBACK_DAYS)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    @classmethod
    def get_rate_limit(cls, api_key: str = None):
        """Get rate limit for given API key"""
        if api_key == 'DEMO_KEY' or api_key is None:
            return cls.RATE_LIMITS['DEMO_KEY']
        return cls.RATE_LIMITS['PERSONAL']