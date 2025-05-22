#!/usr/bin/env python3
"""
Script to download VNP46A2 (5200) data for Myanmar from LAADS DAAC
Product: VNP46A2 - VIIRS/NPP Gap-Filled Lunar BRDF-Adjusted Nighttime Lights Daily L3 Global 500m Linear Lat Lon Grid
Time period: April 1, 2025 to May 20, 2025
"""

import os
import requests
import time
import json
from datetime import datetime, timedelta
from urllib.parse import urljoin
import argparse

class LADSDownloader:
    def __init__(self, api_key=None):
        self.base_url = "https://ladsweb.modaps.eosdis.nasa.gov/api/"
        # You will need to get your own API key from https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#appkeys
        self.api_key = api_key or os.environ.get("LAADS_API_KEY")
        if not self.api_key:
            print("WARNING: No API key provided. You'll need to register for an API key at:")
            print("https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#appkeys")
            print("Then set it with --api-key or as LAADS_API_KEY environment variable")
        
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
    def search_files(self, product="VNP46A2", collection="5200", start_date="2025-04-01", 
                     end_date="2025-05-20", region="Country:MMR"):
        """Search for files based on parameters"""
        search_url = urljoin(self.base_url, "v2/content/archives/allData")
        
        # Construct URL path with parameters
        path = f"{collection}/{product}/{start_date}..{end_date}/DB/{region}/"
        full_url = urljoin(search_url, path)
        
        print(f"Searching for files: {full_url}")
        
        # Make the request
        response = requests.get(full_url, headers=self.headers)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error searching files: {response.status_code}")
            print(response.text)
            return None

    def download_file(self, file_url, output_dir="../../../data/ntl/code_download_daily"):
        """Download a single file"""
        # Extract filename from URL
        filename = os.path.basename(file_url)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Full path for output file
        output_path = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"File already exists: {output_path}")
            return output_path
        
        print(f"Downloading: {filename}")
        response = requests.get(file_url, headers=self.headers, stream=True)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {output_path}")
            return output_path
        else:
            print(f"Error downloading {filename}: {response.status_code}")
            return None

    def download_all_files(self, search_results, output_dir="./data", delay=1):
        """Download all files from search results"""
        if not search_results:
            print("No search results to download")
            return []
        
        downloaded_files = []
        
        for item in search_results:
            # Extract file URL
            file_url = item.get('downloadsLink')
            if file_url:
                result = self.download_file(file_url, output_dir)
                if result:
                    downloaded_files.append(result)
                    # Add delay to avoid overwhelming the server
                    time.sleep(delay)
            else:
                print(f"No download link found for item: {item}")
        
        return downloaded_files

def main():
    parser = argparse.ArgumentParser(description='Download VNP46A2 data from LAADS DAAC')
    parser.add_argument('--api-key', help='LAADS DAAC API key', default='eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InNhaGl0aXNhcnZhIiwiZXhwIjoxNzUwMzk3MzQ1LCJpYXQiOjE3NDUyMTMzNDUsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.NhFQ-k5f3He4CdMog0dbaOu2mGYWA-4zab61DGySmreS0L9eXflBqGCOwNKwWHZmgovvULwjAZY08mVg4Z3cDllrQFFBGkcqAaNPk84jpfmPPC7lGP0zJODrN7oO4V-byxm1R-RkFTTh-SZyxuX_CLzt55jPfTyDhJVdyiTQ_CWDhVjFeuaR0NH0RhWBYG4uoXLD8x7WZdrg299ftbn_FK-Z-MUenUw9A2Vky4QPHf1DgSxMi2HZsPzLwefcOsJrvJu4p9-R6mg3psuq8O8tdg1-fTvwAPEumtEe4WXd53bjDnW-8U6_rFLuXQa6RblDuDmDNjCdlsLzU6LM63dRBQ')
    parser.add_argument('--output-dir', default='./data', help='Directory to save downloaded files')
    parser.add_argument('--product', default='VNP46A2', help='Product to download')
    parser.add_argument('--collection', default='5200', help='Collection number')
    parser.add_argument('--start-date', default='2025-04-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-05-20', help='End date (YYYY-MM-DD)')
    parser.add_argument('--region', default='Country:MMR', help='Region filter')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between downloads in seconds')
    
    args = parser.parse_args()
    
    downloader = LADSDownloader(api_key=args.api_key)
    
    # Search for files
    search_results = downloader.search_files(
        product=args.product,
        collection=args.collection,
        start_date=args.start_date,
        end_date=args.end_date,
        region=args.region
    )
    
    if search_results:
        print(f"Found {len(search_results)} files")
        
        # Download all files
        downloaded = downloader.download_all_files(
            search_results, 
            output_dir=args.output_dir,
            delay=args.delay
        )
        
        print(f"Downloaded {len(downloaded)} files to {args.output_dir}")
    else:
        print("No files found matching criteria")

if __name__ == "__main__":
    main()