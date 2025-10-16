import os
import glob
import re
import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats
import numpy as np
from datetime import datetime

# Paths
tif_dir = "../../../data/ntl/myanmar_nightlights_tif/"  # Directory with TIF files
# admin0_shapefile = "../../../data/boundaries/mmr_polbnda_adm0_250k_mimu_20240215.shp"
# admin1_shapefile = "../../../data/boundaries/mmr_polbnda_adm1_250k_mimu_20240215.shp"
admin2_shapefile = "../../../data/boundaries/mmr_polbnda_adm2_250k_mimu_20240215.shp"
# admin3_shapefile = "../../../data/boundaries/mmr_polbnda_adm3_250k_mimu_20240215.shp"
output_dir = "../../../data/ntl/stats"  # Output directory for statistics

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all TIF files
tif_files = glob.glob(os.path.join(tif_dir, "*.tif"))
print(f"Found {len(tif_files)} TIF files")

# Extract date from filename function
def extract_date_from_filename(filename):
    """Extract date from VNP46A2_Gap_Filled_DNB_BRDF-Corrected_NTL_qflag_t2024_01_07 format"""
    match = re.search(r't(\d{4})_(\d{2})_(\d{2})', os.path.basename(filename))
    if match:
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        try:
            date_obj = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
            return date_obj.strftime("%Y-%m-%d")  # Format as YYYY-MM-DD
        except ValueError:
            return None
    return None

# Extract dates to determine start and end dates for output filenames
all_dates = []
for tif_file in tif_files:
    date_str = extract_date_from_filename(tif_file)
    if date_str:
        all_dates.append(date_str)

if all_dates:
    start_date = min(all_dates)
    end_date = max(all_dates)
    date_range = f"{start_date}_to_{end_date}"
else:
    date_range = "unknown_dates"



# Function to process each admin level
def process_admin_level(shapefile_path, admin_level, admin_column, admin_pcode_column):
    output_stats_path = os.path.join(
        output_dir, 
        f"Myanmar_Admin{admin_level}_Daily_Nightlights_{date_range}.csv"
    )
    
    print(f"\nProcessing Admin{admin_level} level statistics")
    print(f"Using shapefile: {shapefile_path}")
    
    try:
        # Read shapefile
        gdf = gpd.read_file(shapefile_path)[[admin_column, admin_pcode_column, 'geometry']]
        print(f"Shapefile has {len(gdf)} features")
        
        # Calculate zonal statistics for each TIF
        all_results = []
        
        for tif_file in tif_files:
            base_name = os.path.basename(tif_file)
            print(f"Processing {base_name}")
            
            # Extract date
            date_str = extract_date_from_filename(tif_file)
            if not date_str:
                print(f"  Could not extract date from {base_name}, skipping")
                continue
            
            # Extract tile information
            tile_match = re.search(r'h(\d+)[vx](\d+)', base_name)
            tile_str = tile_match.group(0) if tile_match else "unknown"
            
            try:
                # Open and check the raster
                with rasterio.open(tif_file) as src:
                    # Read the data
                    data = src.read(1)
                    
                    # Check for data validity
                    valid_data = data[~np.isnan(data)]
                    if len(valid_data) == 0:
                        print(f"  Skipping {base_name} - no valid data")
                        continue
                    
                    # Check if there are unreasonable negative values
                    if np.any(valid_data < -100):
                        print(f"  Warning: Found very negative values in {base_name}")
                        # Replace extreme negative values with NaN
                        data[data < -100] = np.nan
                        
                    # Create a temporary TIF with fixed data if needed
                    if np.any(data < 0):
                        print(f"  Fixing negative values in {base_name}")
                        # Create a copy with negative values set to 0
                        fixed_data = data.copy()
                        fixed_data[fixed_data < 0] = 0
                        
                        # Write to temp file
                        temp_tif = os.path.join(os.path.dirname(tif_file), f"temp_{os.path.basename(tif_file)}")
                        profile = src.profile
                        with rasterio.open(temp_tif, 'w', **profile) as dst:
                            dst.write(fixed_data, 1)
                        
                        # Use the temp file for zonal stats
                        tif_to_use = temp_tif
                    else:
                        tif_to_use = tif_file
                
                # Calculate statistics using fixed data
                stats = zonal_stats(
                    gdf,
                    tif_to_use,
                    stats=['mean', 'sum', 'count'],
                    nodata=np.nan,
                    all_touched=False
                )
                
                # Clean up temporary file if created
                if 'temp_tif' in locals() and os.path.exists(temp_tif):
                    os.remove(temp_tif)
                
                # Create a record for each feature
                for i, stat in enumerate(stats):
                    if stat['count'] == 0:  # Skip if no valid pixels
                        continue
                        
                    # Ensure sum is not negative
                    if 'sum' in stat and stat['sum'] < 0:
                        stat['sum'] = 0
                    
                    result = {
                        'file': base_name,
                        'feature_id': i,
                        'date': date_str,
                        'tile': tile_str
                    }
                    
                    # Add shapefile attributes
                    for col in gdf.columns:
                        if col != 'geometry':
                            result[col] = gdf.iloc[i][col]
                    
                    # Add statistics
                    for stat_name, value in stat.items():
                        result[stat_name] = value
                    
                    all_results.append(result)
                
                print(f"  Successfully processed with {len(stats)} features")
                
            except Exception as e:
                print(f"  Error processing {base_name}: {e}")
                continue
        
        # Create DataFrame from all individual results
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Drop rows with very small or zero sums (likely no actual lights)
            if 'sum' in results_df.columns:
                non_zero_sum = results_df['sum'] > 1
                print(f"Keeping {non_zero_sum.sum()} of {len(results_df)} rows with sum > 1")
                results_df = results_df[non_zero_sum]
            
            # Aggregate by date and feature (combining all tiles)
            print("Aggregating results by date...")
            
            # Group by date and feature-related columns
            group_cols = ['date', admin_column, admin_pcode_column, 'feature_id']
            
            # Aggregate the statistics
            aggregated_df = results_df.groupby(group_cols).agg({
                'mean': 'mean',  # Average of means weighted by count
                'sum': 'sum',    # Total sum across all tiles
                'count': 'sum',  # Total count of valid pixels
                'tile': lambda x: ','.join(sorted(set(x)))  # List of tiles included
            }).reset_index()
            
            # Calculate weighted mean (if needed for more accuracy)
            if 'count' in results_df.columns:
                # For each group, calculate weighted mean based on pixel counts
                weighted_means = []
                for name, group in results_df.groupby(group_cols):
                    if group['count'].sum() > 0:  # Avoid division by zero
                        # Calculate weighted average: sum(mean * count) / sum(count)                    
                        weighted_mean = (group['mean'] * group['count']).sum() / group['count'].sum()
                    else:
                        weighted_mean = np.nan
                    weighted_means.append(weighted_mean)
                
                # Replace the simple average with the weighted average
                aggregated_df['mean'] = weighted_means
            
            print(f"Reduced from {len(results_df)} to {len(aggregated_df)} rows after aggregation")
            
            # Save aggregated results to CSV
            aggregated_df.to_csv(output_stats_path, index=False)
            print(f"Saved Admin{admin_level} statistics to {output_stats_path}")
            
            # Print a summary
            print(f"\nSummary of Admin{admin_level} output:")
            print(f"Dates: {aggregated_df['date'].nunique()}")
            print(f"Features: {len(aggregated_df['feature_id'].unique())}")
            print(f"Tiles processed: {len(set(','.join(aggregated_df['tile']).split(',')))}")
            if 'sum' in aggregated_df.columns:
                print(f"Sum range: {aggregated_df['sum'].min()} to {aggregated_df['sum'].max()}")
            
            return True
        else:
            print(f"No results were generated for Admin{admin_level}")
            return False
            
    except Exception as e:
        print(f"Error processing Admin{admin_level}: {e}")
        import traceback
        traceback.print_exc()
        return False

# Calculate start and end dates from TIF files
all_dates = []
for tif_file in tif_files:
    date_str = extract_date_from_filename(tif_file)
    if date_str:
        all_dates.append(date_str)

if all_dates:
    start_date = min(all_dates)
    end_date = max(all_dates)
    date_range = f"{start_date}_to_{end_date}"
else:
    date_range = "unknown_dates"

print(f"Date range: {date_range}")

# Process each admin level
# process_admin_level(
#     admin0_shapefile, 
#     0, 
#     'ADM0_EN', 
#     'ADM0_PCODE'
# )

# process_admin_level(
#     admin1_shapefile, 
#     1, 
#     'ADM1_EN', 
#     'ADM1_PCODE'
# )

process_admin_level(
    admin2_shapefile, 
    2, 
    'ADM2_EN', 
    'ADM2_PCODE'
)

# process_admin_level(
#     admin3_shapefile, 
#     2, 
#     'ADM3_EN', 
#     'ADM3_PCODE'
# )

print("All processing complete!")