#!/usr/bin/env python3
"""
Data Analysis Script for NASA DONKI Solar Data
Analyzes and visualizes Solar Flare and CME data
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class DONKIAnalyzer:
    """Analyzes DONKI solar data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Setup matplotlib and seaborn styling"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def load_latest_data(self, data_type: str) -> pd.DataFrame:
        """
        Load the most recent data file
        
        Args:
            data_type: 'flr' for solar flares, 'cme' for CME data
            
        Returns:
            DataFrame with the loaded data
        """
        pattern = f"{self.data_dir}/{data_type}/*.json"
        files = glob.glob(pattern)
        
        if not files:
            print(f"No {data_type} data files found")
            return pd.DataFrame()
        
        # Get the most recent file
        latest_file = max(files, key=os.path.getctime)
        print(f"Loading data from: {latest_file}")
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            print(f"Loaded {len(df)} {data_type} records")
            return df
            
        except Exception as e:
            print(f"Error loading {latest_file}: {e}")
            return pd.DataFrame()
    
    def analyze_solar_flares(self, df: pd.DataFrame):
        """Analyze solar flare data"""
        if df.empty:
            print("No solar flare data to analyze")
            return
        
        print("\n=== SOLAR FLARE ANALYSIS ===")
        
        # Convert time columns to datetime
        time_columns = ['beginTime', 'peakTime', 'endTime']
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Basic statistics
        print(f"Total Solar Flares: {len(df)}")
        if 'classType' in df.columns:
            print(f"Flare Classes: {df['classType'].value_counts().to_dict()}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Solar Flare Analysis', fontsize=16, fontweight='bold')
        
        # 1. Flare frequency over time
        if 'beginTime' in df.columns:
            df_time = df.dropna(subset=['beginTime'])
            if not df_time.empty:
                daily_counts = df_time.set_index('beginTime').resample('D').size()
                axes[0, 0].plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2)
                axes[0, 0].set_title('Daily Solar Flare Count')
                axes[0, 0].set_xlabel('Date')
                axes[0, 0].set_ylabel('Number of Flares')
                axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Flare class distribution
        if 'classType' in df.columns:
            flare_counts = df['classType'].value_counts()
            axes[0, 1].bar(flare_counts.index, flare_counts.values, color='orange', alpha=0.7)
            axes[0, 1].set_title('Flare Class Distribution')
            axes[0, 1].set_xlabel('Flare Class')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Flare duration analysis
        if 'beginTime' in df.columns and 'endTime' in df.columns:
            df_duration = df.dropna(subset=['beginTime', 'endTime'])
            if not df_duration.empty:
                df_duration['duration'] = (df_duration['endTime'] - df_duration['beginTime']).dt.total_seconds() / 3600  # hours
                duration_data = df_duration['duration'].dropna()
                if not duration_data.empty:
                    axes[1, 0].hist(duration_data, bins=20, color='red', alpha=0.7, edgecolor='black')
                    axes[1, 0].set_title('Flare Duration Distribution')
                    axes[1, 0].set_xlabel('Duration (hours)')
                    axes[1, 0].set_ylabel('Frequency')
        
        # 4. Active region analysis
        if 'activeRegionNum' in df.columns:
            ar_counts = df['activeRegionNum'].value_counts().head(10)
            axes[1, 1].bar(range(len(ar_counts)), ar_counts.values, color='purple', alpha=0.7)
            axes[1, 1].set_title('Top 10 Active Regions')
            axes[1, 1].set_xlabel('Active Region Rank')
            axes[1, 1].set_ylabel('Flare Count')
            axes[1, 1].set_xticks(range(len(ar_counts)))
            axes[1, 1].set_xticklabels([str(ar) for ar in ar_counts.index], rotation=45)
        
        plt.tight_layout()
        plt.savefig('solar_flare_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_cme_data(self, df: pd.DataFrame):
        """Analyze CME data"""
        if df.empty:
            print("No CME data to analyze")
            return
        
        print("\n=== CME ANALYSIS ===")
        
        # Convert time columns
        if 'startTime' in df.columns:
            df['startTime'] = pd.to_datetime(df['startTime'], errors='coerce')
        
        print(f"Total CME Events: {len(df)}")
        
        # Extract speed data from cmeAnalyses
        speeds = []
        if 'cmeAnalyses' in df.columns:
            for analyses in df['cmeAnalyses']:
                if isinstance(analyses, list):
                    for analysis in analyses:
                        if isinstance(analysis, dict) and 'speed' in analysis:
                            try:
                                speed = float(analysis['speed'])
                                speeds.append(speed)
                            except (ValueError, TypeError):
                                continue
        
        print(f"CME Speed Records: {len(speeds)}")
        if speeds:
            print(f"Average CME Speed: {sum(speeds)/len(speeds):.1f} km/s")
            print(f"Max CME Speed: {max(speeds):.1f} km/s")
            print(f"Min CME Speed: {min(speeds):.1f} km/s")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Coronal Mass Ejection Analysis', fontsize=16, fontweight='bold')
        
        # 1. CME frequency over time
        if 'startTime' in df.columns:
            df_time = df.dropna(subset=['startTime'])
            if not df_time.empty:
                daily_counts = df_time.set_index('startTime').resample('D').size()
                axes[0, 0].plot(daily_counts.index, daily_counts.values, marker='s', 
                              linewidth=2, color='blue')
                axes[0, 0].set_title('Daily CME Count')
                axes[0, 0].set_xlabel('Date')
                axes[0, 0].set_ylabel('Number of CMEs')
                axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. CME speed distribution
        if speeds:
            axes[0, 1].hist(speeds, bins=30, color='cyan', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('CME Speed Distribution')
            axes[0, 1].set_xlabel('Speed (km/s)')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. CME speed over time
        if speeds and 'startTime' in df.columns:
            # Create a simplified time series for speeds
            speed_data = []
            time_data = []
            for i, analyses in enumerate(df['cmeAnalyses']):
                if isinstance(analyses, list) and analyses:
                    for analysis in analyses:
                        if isinstance(analysis, dict) and 'speed' in analysis:
                            try:
                                speed = float(analysis['speed'])
                                if pd.notna(df.iloc[i]['startTime']):
                                    speed_data.append(speed)
                                    time_data.append(df.iloc[i]['startTime'])
                                    break  # Take first valid speed per CME
                            except (ValueError, TypeError):
                                continue
            
            if speed_data and time_data:
                axes[1, 0].scatter(time_data, speed_data, alpha=0.6, color='red')
                axes[1, 0].set_title('CME Speed Over Time')
                axes[1, 0].set_xlabel('Date')
                axes[1, 0].set_ylabel('Speed (km/s)')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. CME source location analysis
        source_locations = []
        if 'sourceLocation' in df.columns:
            for location in df['sourceLocation'].dropna():
                if location and location.strip():
                    source_locations.append(location)
        
        if source_locations:
            location_counts = pd.Series(source_locations).value_counts().head(10)
            axes[1, 1].barh(range(len(location_counts)), location_counts.values, 
                           color='green', alpha=0.7)
            axes[1, 1].set_title('Top 10 CME Source Locations')
            axes[1, 1].set_xlabel('Count')
            axes[1, 1].set_ylabel('Source Location')
            axes[1, 1].set_yticks(range(len(location_counts)))
            axes[1, 1].set_yticklabels(location_counts.index)
        
        plt.tight_layout()
        plt.savefig('cme_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_combined_analysis(self, flr_df: pd.DataFrame, cme_df: pd.DataFrame):
        """Create combined analysis of solar activity"""
        print("\n=== COMBINED SOLAR ACTIVITY ANALYSIS ===")
        
        if flr_df.empty and cme_df.empty:
            print("No data available for combined analysis")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Combined Solar Activity Timeline', fontsize=16, fontweight='bold')
        
        # Combined timeline
        if not flr_df.empty and 'beginTime' in flr_df.columns:
            flr_time = flr_df.dropna(subset=['beginTime'])
            if not flr_time.empty:
                flr_daily = flr_time.set_index('beginTime').resample('D').size()
                axes[0].plot(flr_daily.index, flr_daily.values, 'r-', 
                           label='Solar Flares', linewidth=2, marker='o')
        
        if not cme_df.empty and 'startTime' in cme_df.columns:
            cme_time = cme_df.dropna(subset=['startTime'])
            if not cme_time.empty:
                cme_daily = cme_time.set_index('startTime').resample('D').size()
                axes[0].plot(cme_daily.index, cme_daily.values, 'b-', 
                           label='CME Events', linewidth=2, marker='s')
        
        axes[0].set_title('Daily Solar Activity Count')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Number of Events')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # Activity correlation
        if (not flr_df.empty and not cme_df.empty and 
            'beginTime' in flr_df.columns and 'startTime' in cme_df.columns):
            
            # Create correlation data
            flr_time = flr_df.dropna(subset=['beginTime'])
            cme_time = cme_df.dropna(subset=['startTime'])
            
            if not flr_time.empty and not cme_time.empty:
                flr_daily = flr_time.set_index('beginTime').resample('D').size()
                cme_daily = cme_time.set_index('startTime').resample('D').size()
                
                # Align the indices
                common_dates = flr_daily.index.intersection(cme_daily.index)
                if len(common_dates) > 1:
                    flr_aligned = flr_daily.reindex(common_dates, fill_value=0)
                    cme_aligned = cme_daily.reindex(common_dates, fill_value=0)
                    
                    axes[1].scatter(flr_aligned.values, cme_aligned.values, 
                                  alpha=0.6, s=50, color='purple')
                    axes[1].set_title('Solar Flare vs CME Activity Correlation')
                    axes[1].set_xlabel('Daily Solar Flares')
                    axes[1].set_ylabel('Daily CME Events')
                    
                    # Add correlation coefficient
                    corr = flr_aligned.corr(cme_aligned)
                    axes[1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                               transform=axes[1].transAxes, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('combined_solar_activity.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        total_flares = len(flr_df) if not flr_df.empty else 0
        total_cmes = len(cme_df) if not cme_df.empty else 0
        
        print(f"Summary Statistics:")
        print(f"- Total Solar Flares: {total_flares}")
        print(f"- Total CME Events: {total_cmes}")
        print(f"- Total Solar Events: {total_flares + total_cmes}")
    
    def generate_report(self, flr_df: pd.DataFrame, cme_df: pd.DataFrame):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE SOLAR ACTIVITY REPORT")
        print("="*60)
        
        # Date range
        dates = []
        if not flr_df.empty and 'beginTime' in flr_df.columns:
            flr_dates = pd.to_datetime(flr_df['beginTime'], errors='coerce').dropna()
            dates.extend(flr_dates)
        
        if not cme_df.empty and 'startTime' in cme_df.columns:
            cme_dates = pd.to_datetime(cme_df['startTime'], errors='coerce').dropna()
            dates.extend(cme_dates)
        
        if dates:
            print(f"Analysis Period: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")
        
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Solar Flare Summary
        print("SOLAR FLARE SUMMARY:")
        print("-" * 20)
        if not flr_df.empty:
            print(f"Total Events: {len(flr_df)}")
            
            if 'classType' in flr_df.columns:
                class_counts = flr_df['classType'].value_counts()
                print("Flare Classifications:")
                for flare_class, count in class_counts.items():
                    print(f"  {flare_class}: {count}")
            
            if 'activeRegionNum' in flr_df.columns:
                active_regions = flr_df['activeRegionNum'].nunique()
                print(f"Active Regions Involved: {active_regions}")
        else:
            print("No solar flare data available")
        
        print()
        
        # CME Summary
        print("CORONAL MASS EJECTION SUMMARY:")
        print("-" * 30)
        if not cme_df.empty:
            print(f"Total Events: {len(cme_df)}")
            
            # Speed analysis
            speeds = []
            if 'cmeAnalyses' in cme_df.columns:
                for analyses in cme_df['cmeAnalyses']:
                    if isinstance(analyses, list):
                        for analysis in analyses:
                            if isinstance(analysis, dict) and 'speed' in analysis:
                                try:
                                    speed = float(analysis['speed'])
                                    speeds.append(speed)
                                except (ValueError, TypeError):
                                    continue
            
            if speeds:
                print(f"Speed Statistics:")
                print(f"  Average: {sum(speeds)/len(speeds):.1f} km/s")
                print(f"  Maximum: {max(speeds):.1f} km/s")
                print(f"  Minimum: {min(speeds):.1f} km/s")
                
                # Categorize by speed
                slow = sum(1 for s in speeds if s < 500)
                medium = sum(1 for s in speeds if 500 <= s < 1000)
                fast = sum(1 for s in speeds if s >= 1000)
                
                print(f"Speed Categories:")
                print(f"  Slow (<500 km/s): {slow}")
                print(f"  Medium (500-1000 km/s): {medium}")
                print(f"  Fast (>1000 km/s): {fast}")
        else:
            print("No CME data available")
        
        print("\n" + "="*60)
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting comprehensive DONKI data analysis...")
        
        # Load data
        flr_df = self.load_latest_data('flr')
        cme_df = self.load_latest_data('cme')
        
        # Run individual analyses
        self.analyze_solar_flares(flr_df)
        self.analyze_cme_data(cme_df)
        
        # Combined analysis
        self.create_combined_analysis(flr_df, cme_df)
        
        # Generate report
        self.generate_report(flr_df, cme_df)
        
        print("\nAnalysis complete! Check the generated PNG files for visualizations.")

def main():
    """Main execution function"""
    analyzer = DONKIAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()