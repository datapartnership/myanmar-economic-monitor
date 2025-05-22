import os
import glob
import re
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from datetime import datetime
from collections import defaultdict

def extract_date_from_filename(filename):
    """Extract date string from VIIRS filename"""
    match = re.search(r'A(\d{7})', os.path.basename(filename))
    if match:
        return match.group(1)
    return None

def parse_tile_coords(tile_str):
    """Extract horizontal and vertical tile coordinates"""
    match = re.search(r'h(\d+)[vx](\d+)', tile_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def calculate_bounds(h, v):
    """Calculate geographic bounds for a tile"""
    left = -180.0 + (h * 10.0)
    top = 90.0 - (v * 10.0)
    right = left + 10.0
    bottom = top - 10.0
    return left, bottom, right, top

def h5_to_geotiff(h5_file, output_dir, dataset_name='Gap_Filled_DNB_BRDF-Corrected_NTL'):
    """Convert an H5 file to GeoTIFF"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract tile info
    base_name = os.path.basename(h5_file)
    tile_match = re.search(r'h\d+[vx]\d+', base_name)
    if not tile_match:
        print(f"Could not extract tile info from {base_name}, skipping")
        return None
    
    tile_str = tile_match.group(0)
    h, v = parse_tile_coords(tile_str)
    
    # Calculate bounds
    left, bottom, right, top = calculate_bounds(h, v)
    
    # Open H5 file and extract data
    try:
        with h5py.File(h5_file, 'r') as f:
            # Get the dataset
            data = f['HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields'][dataset_name][:]
            
            # Get scale factor and fill value
            try:
                scale_factor = f['HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields'][dataset_name].attrs['scale_factor']
            except:
                scale_factor = 0.1  # Default value
                
            try:
                fill_value = f['HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields'][dataset_name].attrs['_FillValue']
            except:
                fill_value = 65535  # Default value
    except Exception as e:
        print(f"Error reading {h5_file}: {e}")
        return None
    
    # Process data
    data = data.astype(np.float32)
    data[data == fill_value] = np.nan
    data = data * scale_factor
    
    # Calculate resolution
    xres = (right - left) / data.shape[1]
    yres = (top - bottom) / data.shape[0]
    
    # Create transform
    transform = from_origin(left, top, xres, yres)
    
    # Output filename
    output_file = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.tif")
    
    # Write GeoTIFF
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(data, 1)
    
    print(f"Converted {h5_file} to {output_file}")
    return output_file

def apply_zonal_stats(tif_files, shapefile, output_file, stats=None):
    """Apply zonal statistics to the TIF files"""
    if stats is None:
        stats = ['min', 'max', 'mean', 'median', 'sum', 'std', 'count']
    
    # Read shapefile
    gdf = gpd.read_file(shapefile)
    
    # Create output dataframe to store all results
    results_gdf = gdf.copy()
    
    # Process each TIF file
    for tif_file in tif_files:
        if tif_file is None:
            continue
            
        base_name = os.path.basename(tif_file)
        print(f"Processing zonal statistics for {base_name}")
        
        # Calculate zonal statistics
        zone_stats = zonal_stats(
            shapefile,
            tif_file,
            stats=stats,
            geojson_out=True,
            all_touched=False
        )
        
        # Add statistics to the GeoDataFrame
        for stat in stats:
            col_name = f"{os.path.splitext(base_name)[0]}_{stat}"
            results_gdf[col_name] = [feat['properties'][stat] for feat in zone_stats]
    
    # Save results
    # Save as CSV (without geometry)
    results_gdf.drop('geometry', axis=1).to_csv(f"{output_file}.csv", index=False)
    
    # Save as GeoJSON (with geometry)
    results_gdf.to_file(f"{output_file}.geojson", driver="GeoJSON")
    
    print(f"Saved zonal statistics to {output_file}.csv and {output_file}.geojson")
    
    return results_gdf

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process VIIRS H5 files for a single day to TIFs and perform zonal statistics')
    parser.add_argument('--input_dir', '-i', required=True, help='Directory containing H5 files')
    parser.add_argument('--date', '-d', required=True, help='Date to process in format YYYYDDD (e.g., 2025139)')
    parser.add_argument('--output_dir', '-o', default='output', help='Output directory')
    parser.add_argument('--dataset', default='Gap_Filled_DNB_BRDF-Corrected_NTL', help='Dataset name to extract')
    parser.add_argument('--shapefile', '-s', required=True, help='Shapefile for zonal statistics')
    parser.add_argument('--output_stats', default='zonal_stats_results', help='Output filename for statistics (without extension)')
    parser.add_argument('--stats', default='min,max,mean,median,sum,std,count', help='Statistics to calculate')
    
    args = parser.parse_args()
    
    # Find all H5 files for the specified date
    date_pattern = f"*A{args.date}*.h5"
    h5_files = glob.glob(os.path.join(args.input_dir, date_pattern))
    
    if not h5_files:
        print(f"No H5 files found for date {args.date}")
        return
    
    print(f"Found {len(h5_files)} H5 files for date {args.date}")
    
    # Convert H5 files to GeoTIFF
    tif_dir = os.path.join(args.output_dir, 'tiffs')
    os.makedirs(tif_dir, exist_ok=True)
    
    tif_files = []
    for h5_file in h5_files:
        tif_file = h5_to_geotiff(h5_file, tif_dir, args.dataset)
        if tif_file:
            tif_files.append(tif_file)
    
    if not tif_files:
        print("No TIF files were created successfully")
        return
    
    # Apply zonal statistics
    stats = args.stats.split(',')
    results = apply_zonal_stats(tif_files, args.shapefile, os.path.join(args.output_dir, args.output_stats), stats)
    
    print("Processing complete")

if __name__ == "__main__":
    main()